#include "kernels.cuh"



Image heightfield;

typedef struct
{
    float4 m[4];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

struct Ray
{
    float3 o;   // origin
    float3 d;   // direction
};

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4& M, const float3& v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4& M, const float4& v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float* tnear, float* tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

extern "C"
void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}
extern "C"
void freeCudaBuffers()
{
    checkCudaErrors(cudaDestroyTextureObject(heightfield.textureObject));
    checkCudaErrors(cudaFreeArray(heightfield.dataArray));
    checkCudaErrors(cudaFreeMipmappedArray(heightfield.mipmapArray));
}

//raymarchkernel << < grid, block >> > (d_output, window_width, window_height);
extern "C" void raymarchkernelStart(uchar4 * d_output, unsigned int imageW, unsigned int imageH, dim3 grid, dim3 block) {
    cudaExtent size = heightfield.size;
    int3 size1 = { size.width,size.height,size.depth };
    float height = .15f;
    raymarchkernel << < grid, block >> > (d_output, imageW, imageH,heightfield.textureObject, size1, height);
}

__device__ const float eps = 0.0001f;

__device__
float MarchRay(Ray r, cudaTextureObject_t tex, int3 size, int maxIter = 128, float height=0.1f) {
    //r.o = (r.o + 1.0f) * 0.5f * make_float3(size.x, size.y, 1.0f / height);
    float3 conversion =  make_float3(size.x, size.y, 1.0f) / make_float3(2.0f, 2.0f, height);
    r.o = (r.o + make_float3(1.0f,1.0f,0.0f)) * conversion;
    r.d = r.d * conversion;
    float3 r0 = r.o;
    float2 xlim = { 0.0f, float(size.x) };
    float2 ylim = { 0.0f, float(size.y) };
    float dist = -1;
    float TotalDist = 0;
    int CurrentMip = size.z;
    int count = 0;
    int TotalSteps = 0;
    for (int i = 0; i < maxIter; ++i) {
        if (CurrentMip == -1 || CurrentMip == size.z + 1) break;
        
        if (r.o.x < eps || r.o.x > size.x - eps || r.o.y < eps || r.o.y > size.y - eps
            || r.o.x == 0.0f && r.d.x < 0.0f || r.o.x == size.x && r.d.x > 0.0f
            || r.o.y == 0.0f && r.d.y < 0.0f || r.o.y == size.y && r.d.y > 0.0f
            || r.o.z >= 1.0f - eps && r.d.z > 0.0f)
            return -1.0f;
        // луч выходит за пределы

        bool above = false;
        float Z;

        for (; above == false && CurrentMip >= 0; --CurrentMip) {
            float mippow = float(1 << CurrentMip);
            float x = floor(r.o.x / mippow);
            float y = floor(r.o.y / mippow);

            xlim = { x * mippow, (x + 1.0f) * mippow };
            ylim = { y * mippow, (y + 1.0f) * mippow };
            
            
            if (r.o.x == xlim.x && r.d.x < 0) {
                xlim -= mippow;
                x -= 1.0f;
            }
            if (r.o.y == ylim.x && r.d.y < 0) {
                ylim -= mippow;
                y -= 1.0f;
            }

            x = fminf(fmaxf(x, 0.0f), size.x - 1);
            y = fminf(fmaxf(y, 0.0f), size.y - 1);
            Z = (tex2DLod<float4>(tex, (x + 0.5f) * mippow / size.x, (y + 0.5f) * mippow / size.y, CurrentMip)).w;
            //Z = (tex2DLod<float4>(tex, r.o.x / size.x, r.o.y / size.y, CurrentMip)).w;
            above = (Z < r.o.z);
            //if (above == false) CurrentMip = CurrentMip - 1;
        }

        if (above == false) break;
        else ++CurrentMip;//return TotalDist;

        TotalSteps = TotalSteps + 1;

        float z1 = (-r.o.z + Z) / r.d.z;
        float z2 = (-r.o.z + 1.0f) / r.d.z;
        float x1 = (-r.o.x + xlim.x) / r.d.x;
        float x2 = (-r.o.x + xlim.y) / r.d.x;
        float y1 = (-r.o.y + ylim.x) / r.d.y;
        float y2 = (-r.o.y + ylim.y) / r.d.y;
        x1 = max(x1, x2);
        y1 = max(y1, y2);
        float z3 = max(max(z1, z2), 0.0f);
        dist = min(min(x1, y1), z3);
        r.o = r.o + dist * r.d;
        TotalDist = TotalDist + dist;

        //if (r.o.z <= Z + eps || dist == z1) { // ход вглубь, если луч пересекает Z грань или ниже ее
        if (dist == z1) {
            CurrentMip = CurrentMip - 1;
            count = 1;
        }
        else {//% ход в сторону и наружу
            if (count % 3 == 0)
                CurrentMip = CurrentMip + 1;
            count = count + 1;
        }
    }
    float3 tmp = (r0 - r.o) / conversion;
    return sqrtf(dot(tmp, tmp));//TotalDist;
    //return float(TotalSteps);
}


__global__ void raymarchkernel(uchar4* d_output, unsigned int imageW, unsigned int imageH, cudaTextureObject_t tex, int3 size, float height)
{
    //if (tex == NULL) tex = heightfield.textureObject;
    const int maxSteps = 0;// 500;
    const float tstep = 0.01f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, 0.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, height);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float)imageW) * 2.0f - 1.0f;
    float v = (y / (float)imageH) * 2.0f - 1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;
    int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);

    if (!hit) return;

    if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d * tnear;
    float3 step = eyeRay.d * tstep;

    eyeRay.o = pos + eps * eyeRay.d;
    float dist = MarchRay(eyeRay, tex, size, 128, height);
    if (dist < 0) return;
    pos = eyeRay.o + dist * eyeRay.d;

    //sum *= brightness;

    // write output color
    sum = 255*tex2DLod<float4>(tex, pos.x * 0.5f + 0.5f, pos.y * 0.5f + 0.5f, 0.0f);
    sum.w = 255.0f;
    //sum = make_float4(dist, dist, dist,1.0f);
    d_output[y * imageW + x] = to_uchar4(sum);
}

float highestLod = 1.0f;

__global__ void
d_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH, bool modeMax = false)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    float px = 1.0 / float(imageW);
    float py = 1.0 / float(imageH);


    if ((x < imageW) && (y < imageH))
    {
        // take the average of 4 samples

        // we are using the normalized access to make sure non-power-of-two textures
        // behave well when downsized.
        float4 color;
        //if (modeMax) color =
            //max(max((tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)),
            //(tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py))),
            //max((tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)),
            //(tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py))));
        //else
        color =
            (tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py)) +
            (tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)) +
            (tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py));
        color /= 4.0;
        
        if (modeMax)
            color.w = max(max((tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)).w,
            (tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py)).w),
            max((tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)).w,
            (tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py)).w));
        
        color *= 255.0;
        color = fminf(color, make_float4(255.0));

        surf2Dwrite(to_uchar4(color), mipOutput, x * sizeof(uchar4), y);
    }
}

void generateMipMaps(cudaMipmappedArray_t mipmapArray, cudaExtent size, bool modeMax = false)
{
    size_t width = size.width;
    size_t height = size.height;

#ifdef SHOW_MIPMAPS
    cudaArray_t levelFirst;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFirst, mipmapArray, 0));
#endif

    uint level = 0;

    while (width != 1 || height != 1)
    {
        width /= 2;
        width = MAX((size_t)1, width);
        height /= 2;
        height = MAX((size_t)1, height);

        cudaArray_t levelFrom;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelFrom, mipmapArray, level));
        cudaArray_t levelTo;
        checkCudaErrors(cudaGetMipmappedArrayLevel(&levelTo, mipmapArray, level + 1));

        cudaExtent  levelToSize;
        checkCudaErrors(cudaArrayGetInfo(NULL, &levelToSize, NULL, levelTo));
        //checkHost(levelToSize.width == width);
        //checkHost(levelToSize.height == height);
        //checkHost(levelToSize.depth == 0);

        // generate texture object for reading
        cudaTextureObject_t         texInput;
        cudaResourceDesc            texRes;
        memset(&texRes, 0, sizeof(cudaResourceDesc));

        texRes.resType = cudaResourceTypeArray;
        texRes.res.array.array = levelFrom;

        cudaTextureDesc             texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = 1;
        texDescr.filterMode = cudaFilterModeLinear;

        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;

        texDescr.readMode = cudaReadModeNormalizedFloat;

        checkCudaErrors(cudaCreateTextureObject(&texInput, &texRes, &texDescr, NULL));

        // generate surface object for writing

        cudaSurfaceObject_t surfOutput;
        cudaResourceDesc    surfRes;
        memset(&surfRes, 0, sizeof(cudaResourceDesc));
        surfRes.resType = cudaResourceTypeArray;
        surfRes.res.array.array = levelTo;

        checkCudaErrors(cudaCreateSurfaceObject(&surfOutput, &surfRes));

        // run mipmap kernel
        dim3 blockSize(16, 16, 1);
        dim3 gridSize(((uint)width + blockSize.x - 1) / blockSize.x, ((uint)height + blockSize.y - 1) / blockSize.y, 1);

        d_mipmap << <gridSize, blockSize >> > (surfOutput, texInput, (uint)width, (uint)height, modeMax);

        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaDestroySurfaceObject(surfOutput));

        checkCudaErrors(cudaDestroyTextureObject(texInput));

#ifdef SHOW_MIPMAPS
        // we blit the current mipmap back into first level
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.dstArray = levelFirst;
        copyParams.srcArray = levelTo;
        copyParams.extent = make_cudaExtent(width, height, 1);
        copyParams.kind = cudaMemcpyDeviceToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
#endif

        level++;
    }
}

uint getMipMapLevels(cudaExtent size)
{
    size_t sz = MAX(MAX(size.width, size.height), size.depth);

    uint levels = 0;

    while (sz)
    {
        sz /= 2;
        levels++;
    }

    return levels;
}

extern "C"
void initCuda(Image& image_in)
{
    Image& image = heightfield;
    image.size = image_in.size;
    image.size.depth = 0;
    image.type = cudaResourceTypeMipmappedArray;
    // how many mipmaps we need
    uint levels = getMipMapLevels(image.size);
    highestLod = MAX(highestLod, (float)levels - 1);

    // create 2D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    //checkCudaErrors(cudaMallocMipmappedArray(&d_heightfieldArray, &channelDesc, image.size, levels));
    checkCudaErrors(cudaMallocMipmappedArray(&image.mipmapArray, &channelDesc, image.size, levels));

    cudaArray_t level0;
    checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, image.mipmapArray, 0));
    // загрузка в память нулевого мипа
    checkCudaErrors(cudaMemcpyToArray(level0,0,0, image_in.h_data, image.size.width * image.size.height * sizeof(uchar4),cudaMemcpyHostToDevice));

    // compute rest of mipmaps based on level 0
    generateMipMaps(image.mipmapArray, image.size, true);

    // generate bindless texture object

    cudaResourceDesc            resDescr;
    memset(&resDescr, 0, sizeof(cudaResourceDesc));

    resDescr.resType = cudaResourceTypeMipmappedArray;
    resDescr.res.mipmap.mipmap = image.mipmapArray;

    cudaTextureDesc             texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = 1;
    //texDescr.filterMode = cudaFilterModeLinear;
    texDescr.filterMode = cudaFilterModePoint;
    //texDescr.mipmapFilterMode = cudaFilterModeLinear;
    texDescr.mipmapFilterMode = cudaFilterModePoint;

    texDescr.addressMode[0] = cudaAddressModeClamp;
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;

    texDescr.maxMipmapLevelClamp = float(levels - 1);

    texDescr.readMode = cudaReadModeNormalizedFloat;

    checkCudaErrors(cudaCreateTextureObject(&image.textureObject, &resDescr, &texDescr, NULL));
    image.size.depth = (int)highestLod;

}