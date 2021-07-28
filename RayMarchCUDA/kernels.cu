#include "kernels.cuh"
#include "raysgather.h"

Image heightfield;
//ImageCube scene;
cubetex sceneCube;

std::vector<Image>  scene;

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
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float &tnear, float &tfar)
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

    tnear = largest_tmin;
    tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

extern "C"
void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}

/*
extern "C"
void freeCudaBuffers()
{
    checkCudaErrors(cudaDestroyTextureObject(heightfield.textureObject));
    checkCudaErrors(cudaFreeArray(heightfield.dataArray));
    checkCudaErrors(cudaFreeMipmappedArray(heightfield.mipmapArray));
}
*/
extern "C"
void freeCudaBuffers()
{
    for (size_t i = 0; i < scene.size(); i++)
    {
        Image& image = scene[i];

        if (image.h_data)
        {
            free(image.h_data);
        }

        if (image.textureObject)
        {
            checkCudaErrors(cudaDestroyTextureObject(image.textureObject));
        }

        if (image.mipmapArray)
        {
            checkCudaErrors(cudaFreeMipmappedArray(image.mipmapArray));
        }
    }
}

#define MAX_REFL 8

//raymarchkernel << < grid, block >> > (d_output, window_width, window_height);
extern "C" void raymarchkernelStart(uchar4 * d_output, unsigned int imageW, unsigned int imageH, dim3 grid, dim3 block, float4* IR, int intensitySetting, int mode) {
    cudaExtent size = scene[0].size;// heightfield.size;
    int3 size1 = { size.width,size.height,size.depth };
    float brightness = pow(1.2f, intensitySetting);
    //float height = 2.0f;// .15f;
    //raymarchkernel << < grid, block >> > (d_output, imageW, imageH,heightfield.textureObject, size1, height);
    //raymarchkernel <<< grid, block >>> (d_output, imageW, imageH, sceneCube, size1);
    //raymarchkernelMultipleReflections << < grid, block, block.x * block.y * MAX_REFL * sizeof(float4) >> > (d_output, imageW, imageH, sceneCube, size1);
    //raymarchkernelMultipleReflections << < grid, block >> > (d_output, imageW, imageH, sceneCube, size1, brightness);
    float4* d_IR = NULL;
    int side = 512;

    if (mode == 1) {
        raymarchkernelMultipleReflections << < grid, block >> > (d_output, imageW, imageH, sceneCube, size1, brightness);
    }
    if (mode == 2) {
        int NumberElements = imageW * imageH;
        cudaMalloc((void**)&d_IR, NumberElements * 2 * sizeof(float4));
        raymarchkernelSound << < grid, block >> > (d_output, imageW, imageH, sceneCube, size1, d_IR, brightness);
        cudaFree(d_IR);
    } 
    if (mode == 3) {
        int NumberElements = side * side * 6;
        cudaMalloc((void**)&d_IR, NumberElements * 2 * sizeof(float4));
        dim3 grid1(side / block.x, side / block.x, 6);
        raymarchkernelSoundCube << < grid1, block >> > (d_output, side, imageW, imageH, sceneCube, size1, d_IR, brightness);

        //IR = new float4[NumberElements * 2];
        //float4* IR_out = new float4[NumberElements * 2];
        cudaDeviceSynchronize();
        //cudaMemcpy(IR, d_IR, imageW * imageH * 2 * sizeof(float4), cudaMemcpyDeviceToHost);
        //cudaMemcpy(IR, d_IR, NumberElements * 2 * sizeof(float4), cudaMemcpyDeviceToHost);
        cudaFree(d_IR);
        //int M = gatherRays(IR, IR_out, NumberElements);
        //std::cout << NumberElements << " " << M << std::endl;
        //delete IR_out;
        //delete IR;
    }
    
}

__device__ const float eps = 0.0001f;


__device__ 
float4 sample(cubetex tex, int3 size, float3 v, float l) {
    //return tex2DLod<float4>(tex.px, v.x , v.y , l);
    float4 val = make_float4(0.0f);
    float2 uv;
    //int face=0;
    cudaTextureObject_t texf = tex.mz;
    if (abs(v.x) > abs(v.y) && abs(v.x) > abs(v.z)) {
        if (v.x >= 0.0f) { /*face = 0;*/ v /= v.x; uv.x = v.z; uv.y = v.y; texf = tex.px; } //val = make_float4(1.0f,0.0f,0.0f,1.0f); }
        else          { /*face = 1;*/ v /= -v.x; uv.x = -v.z; uv.y = v.y; texf = tex.mx; } //val = make_float4(0.0f,0.0f,0.0f,1.0f); }
    } 
    else if (abs(v.y) > abs(v.x) && abs(v.y) > abs(v.z)) {
        if (v.y >= 0.0f) {/*face = 2;*/  v /= v.y; uv.x = v.z; uv.y = -v.x; texf = tex.py; } //val = make_float4(0.0f,1.0f,0.0f,1.0f); }
        else          { /*face = 3;*/ v /= -v.y; uv.x = v.z; uv.y = v.x; texf = tex.my; } //val = make_float4(0.0f,0.0f,0.0f,1.0f); }
    }
    else if (abs(v.z) > abs(v.x) && abs(v.z) > abs(v.y)) {
        if (v.z >= 0.0f) { /*face = 4;*/ v /= v.z; uv.x = -v.x; uv.y = v.y; texf = tex.pz; } //val = make_float4(0.0f,0.0f,1.0f,1.0f); }
        else          { /*face = 5;*/ v /= -v.z; uv.x = v.x; uv.y = v.y; texf = tex.mz; } //val = make_float4(0.0f,0.0f,0.0f,1.0f); }
    }
    uv.y *= -1.0f;
    uv = (uv + 1.0f)*0.5f;
    if (isnan(uv.x) || isnan(uv.y) || isinf(uv.x) || isinf(uv.y)) return val;
    val = tex2DLod<float4>(texf, uv.x, uv.y, l);
    return val;
    //return make_float4(uv.x,uv.y,1.0f,1.0f);
}

/*
__device__
float4 sample(cubetex tex, int3 size, float3 v, float l) {
    //return tex2DLod<float4>(tex.px, v.x , v.y , l);
    float4 val = make_float4(0.0f);
    float2 uv;
    int face = 0;
    cudaTextureObject_t texf = tex.mz;
    if (v.x >= abs(v.y) && v.x >= abs(v.z))
        { face = 0; v /= v.x; uv.x = v.z; uv.y = v.y; texf = tex.px; }
    else if (-v.x >= abs(v.y) && -v.x >= abs(v.z))
        { face = 1; v /= -v.x; uv.x = -v.z; uv.y = v.y; texf = tex.mx; }
    else if (v.y >= abs(v.x) && v.y >= abs(v.z))
        { face = 2; v /= v.y; uv.x = v.z; uv.y = -v.x; texf = tex.py; }
    else if (-v.y >= abs(v.x) && -v.y >= abs(v.z)) 
        { face = 3; v /= -v.y; uv.x = v.z; uv.y = v.x; texf = tex.my; }
    else if (v.z >= abs(v.x) && v.z >= abs(v.y))
        { face = 4; v /= v.z; uv.x = -v.x; uv.y = v.y; texf = tex.pz; }
    else if (-v.z >= abs(v.x) && -v.z >= abs(v.y)) 
        { face = 5; v /= -v.z; uv.x = v.x; uv.y = v.y; texf = tex.mz; }
    uv.y *= -1.0f;
    uv = (uv + 1.0f) * 0.5f;
    if (isnan(uv.x) || isnan(uv.y) || isinf(uv.x) || isinf(uv.y)) return val;
    val = tex2DLod<float4>(texf, uv.x, uv.y, l);
    return val;
    //return make_float4(uv.x,uv.y,1.0f,1.0f);
}
*/
__device__
float4 sampleNormals(cubetex tex, int3 size, float3 v, float l) {
    //return tex2DLod<float4>(tex.px, v.x , v.y , l);
    float4 val = make_float4(0.0f);
    float2 uv;
    int face=0;
    cudaTextureObject_t texf = tex.mz;
    if (abs(v.x) > abs(v.y) && abs(v.x) > abs(v.z)) {
        if (v.x >= 0.0f) { face = 0; v /= v.x; uv.x = v.z; uv.y = v.y; texf = tex.N_px; } //val = make_float4(1.0f,0.0f,0.0f,1.0f); }
        else          { face = 1; v /= -v.x; uv.x = -v.z; uv.y = v.y; texf = tex.N_mx; } //val = make_float4(0.0f,0.0f,0.0f,1.0f); }
    } 
    else if (abs(v.y) > abs(v.x) && abs(v.y) > abs(v.z)) {
        if (v.y >= 0.0f) { face = 2; v /= v.y; uv.x = v.z; uv.y = -v.x; texf = tex.N_py; } //val = make_float4(0.0f,1.0f,0.0f,1.0f); }
        else          { face = 3; v /= -v.y; uv.x = v.z; uv.y = v.x; texf = tex.N_my; } //val = make_float4(0.0f,0.0f,0.0f,1.0f); }
    }
    else if (abs(v.z) > abs(v.x) && abs(v.z) > abs(v.y)) {
        if (v.z >= 0.0f) { face = 4; v /= v.z; uv.x = -v.x; uv.y = v.y; texf = tex.N_pz; } //val = make_float4(0.0f,0.0f,1.0f,1.0f); }
        else          { face = 5; v /= -v.z; uv.x = v.x; uv.y = v.y; texf = tex.N_mz; } //val = make_float4(0.0f,0.0f,0.0f,1.0f); }
    }
    uv.y *= -1.0f;
    uv = (uv + 1.0f)*0.5f;
    if (isnan(uv.x) || isnan(uv.y) || isinf(uv.x) || isinf(uv.y)) return val;
    val = tex2DLod<float4>(texf, uv.x, uv.y, l);
    //if (isnan(val.x) || isnan(val.y) || isnan(val.z) || isnan(val.w) || isinf(val.x) || isinf(val.y) || isinf(val.z) || isinf(val.w)) val = make_float4(0.0f);
    return val;
    //return make_float4(uv.x,uv.y,1.0f,1.0f);
}



__device__ inline float Z2world(float x) { return 1.0f / x; }

__device__ inline float3 world2face(float3 v, int face) { 
    float3 uv;
    /*
    if      (face == 0) { uv.z = v.x; v /= v.x; uv.x = v.z; uv.y = v.y;  }
    else if (face == 1) { uv.z = -v.x; v /= -v.x; uv.x = -v.z; uv.y = v.y; }
    else if (face == 2) { uv.z = v.y; v /= v.y; uv.x = v.z; uv.y = -v.x; }
    else if (face == 3) { uv.z = -v.y; v /= -v.y; uv.x = v.z; uv.y = v.x; }
    else if (face == 4) { uv.z = v.z; v /= v.z; uv.x = -v.x; uv.y = v.y; }
    else if (face == 5) { uv.z = -v.z; v /= -v.z; uv.x = v.x; uv.y = v.y; }
    */
    if      (face == 0) { uv.z = Z2world(v.x); v /= abs(v.x); uv.x = v.z; uv.y = v.y; }
    else if (face == 1) { uv.z = Z2world(-v.x); v /= abs(v.x); uv.x = -v.z; uv.y = v.y; }
    else if (face == 2) { uv.z = Z2world(v.y); v /= abs(v.y); uv.x = v.z; uv.y = -v.x; }
    else if (face == 3) { uv.z = Z2world(-v.y); v /= abs(v.y); uv.x = v.z; uv.y = v.x; }
    else if (face == 4) { uv.z = Z2world(v.z); v /= abs(v.z); uv.x = -v.x; uv.y = v.y; }
    else if (face == 5) { uv.z = Z2world(-v.z); v /= abs(v.z); uv.x = v.x; uv.y = v.y; }
    uv.y *= -1.0f;
    //uv.z *= -1.0f;
    //uv.x += 1.0f; uv.y += 1.0f; uv.x *= 0.5f; uv.y *= 0.5f;
    //uv.x *= 1024.0f; uv.y *= 1024.0f;
    return uv;
}
__device__ inline float3 face2world(float3 uv, int face) {
    //uv.x /= 1024.0f; uv.y /= 1024.0f;
    //uv.x *= 2.0f; uv.y *= 2.0f; uv.x -= 1.0f; uv.y -= 1.0f;
    uv.y *= -1.0f;
    //uv.z *= -1.0f;
    float3 v = make_float3(1.0f);
    
    //if (face == 0)      { v.z = uv.x ; v.y = uv.y; /*v.x = uv.z;*/  v *= uv.z; }
    //else if (face == 1) { v.z = -uv.x; v.y = uv.y; /*v.x = -uv.z;*/  v *= -uv.z; }
    //else if (face == 2) { v.z = uv.x; v.x = -uv.y; /*v.y = uv.z;*/  v *= uv.z; }
    //else if (face == 3) { v.z = uv.x; v.x = uv.y;  /*v.y = -uv.z;*/  v *= -uv.z; }
    //else if (face == 4) { v.x = -uv.x; v.y = uv.y; /*v.z = uv.z;*/  v *= uv.z; }
    //else if (face == 5) { v.x = uv.x; v.y = uv.y;  /*v.z = -uv.z;*/  v *= -uv.z; }
    
    if      (face == 0) { v.z = uv.x; v.y = uv.y; v.x = 1.0f;  v *= abs(Z2world(uv.z)); }
    else if (face == 1) { v.z = -uv.x; v.y = uv.y; v.x = -1.0f;  v *= abs(Z2world(uv.z)); }
    else if (face == 2) { v.z = uv.x; v.x = -uv.y; v.y = 1.0f;  v *= abs(Z2world(uv.z)); }
    else if (face == 3) { v.z = uv.x; v.x = uv.y;  v.y = -1.0f;  v *= abs(Z2world(uv.z)); }
    else if (face == 4) { v.x = -uv.x; v.y = uv.y; v.z = 1.0f;  v *= abs(Z2world(uv.z)); }
    else if (face == 5) { v.x = uv.x; v.y = uv.y;  v.z = -1.0f;  v *= abs(Z2world(uv.z)); }
    return v;
}

__device__ inline float3 dir2face(float3 v, int face) {
    float3 uv;
    if      (face == 0) { uv.z = v.x; uv.x = v.z; uv.y = v.y; }
    else if (face == 1) { uv.z = -v.x; uv.x = -v.z; uv.y = v.y; }
    else if (face == 2) { uv.z = v.y; uv.x = v.z; uv.y = -v.x; }
    else if (face == 3) { uv.z = -v.y; uv.x = v.z; uv.y = v.x; }
    else if (face == 4) { uv.z = v.z; uv.x = -v.x; uv.y = v.y; }
    else if (face == 5) { uv.z = -v.z; uv.x = v.x; uv.y = v.y; }
    uv.y *= -1.0f;
    //uv.z *= -1.0f;
    //uv.x += 1.0f; uv.y += 1.0f; uv.x *= 0.5f; uv.y *= 0.5f;
    //uv.x *= 1024.0f; uv.y *= 1024.0f;
    return uv;
}

__device__ float4 sampleNormalsAndDepth(cubetex tex, int3 size, float3 v, float l, float& depth) {
    //return tex2DLod<float4>(tex.px, v.x , v.y , l);
    float4 val = make_float4(0.0f);
    float2 uv;
    int face = 0;
    cudaTextureObject_t texf = tex.mz;
    if (abs(v.x) > abs(v.y) && abs(v.x) > abs(v.z)) {
        if (v.x >= 0.0f) { face = 0; v /= v.x; uv.x = v.z; uv.y = v.y; texf = tex.N_px; } //val = make_float4(1.0f,0.0f,0.0f,1.0f); }
        else { face = 1; v /= -v.x; uv.x = -v.z; uv.y = v.y; texf = tex.N_mx; } //val = make_float4(0.0f,0.0f,0.0f,1.0f); }
    }
    else if (abs(v.y) > abs(v.x) && abs(v.y) > abs(v.z)) {
        if (v.y >= 0.0f) { face = 2; v /= v.y; uv.x = v.z; uv.y = -v.x; texf = tex.N_py; } //val = make_float4(0.0f,1.0f,0.0f,1.0f); }
        else { face = 3; v /= -v.y; uv.x = v.z; uv.y = v.x; texf = tex.N_my; } //val = make_float4(0.0f,0.0f,0.0f,1.0f); }
    }
    else if (abs(v.z) > abs(v.x) && abs(v.z) > abs(v.y)) {
        if (v.z >= 0.0f) { face = 4; v /= v.z; uv.x = -v.x; uv.y = v.y; texf = tex.N_pz; } //val = make_float4(0.0f,0.0f,1.0f,1.0f); }
        else { face = 5; v /= -v.z; uv.x = v.x; uv.y = v.y; texf = tex.N_mz; } //val = make_float4(0.0f,0.0f,0.0f,1.0f); }
    }
    uv.y *= -1.0f;
    depth = norm3df(1.0f, uv.x, uv.y);
    uv = (uv + 1.0f) * 0.5f;
    if (isnan(uv.x) || isnan(uv.y) || isinf(uv.x) || isinf(uv.y)) return val;
    val = tex2DLod<float4>(texf, uv.x, uv.y, l);
    depth *= Z2world(val.w);
    //if (isnan(val.x) || isnan(val.y) || isnan(val.z) || isnan(val.w) || isinf(val.x) || isinf(val.y) || isinf(val.z) || isinf(val.w)) val = make_float4(0.0f);
    return val;
    //return make_float4(uv.x,uv.y,1.0f,1.0f);
}

__device__ inline int getFaceNumber(float3 r, cudaTextureObject_t &texf, cubetex tex) {
    int face;// = 5;
    face = 5;
    if (abs(r.x) > abs(r.y) && abs(r.x) > abs(r.z)) {
        if (r.x >= 0.0f) { face = 0; texf = tex.N_px; }
        else { face = 1; texf = tex.N_mx; }
    }
    else if (abs(r.y) > abs(r.x) && abs(r.y) > abs(r.z)) {
        if (r.y >= 0.0f) { face = 2; texf = tex.N_py; }
        else { face = 3; texf = tex.N_my; }
    }
    else if (abs(r.z) > abs(r.x) && abs(r.z) > abs(r.y)) {
        if (r.z >= 0.0f) { face = 4; texf = tex.N_pz; }
        else { face = 5; texf = tex.N_mz; }
    }
    return face;
}

__device__ float2 intersectSphere(Ray r, float3 sphere_pos, float radius) {
    float3 oc = r.o - sphere_pos;
    float b = dot(oc, r.d);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) return make_float2(-1.0f); // no intersection
    h = sqrt(h);
    return make_float2(-b - h, -b + h);
}

__device__
float2 MarchRay(Ray r, cubetex tex, int3 size, int maxIter = 128, int levelOfDetails = 0) {
    r.o = r.o + eps * r.d;
    float3 r0 = r.o;
    float2 xlim = { 0.0f, float(size.x) };
    float2 ylim = { 0.0f, float(size.y) };
    float dist = -1.0f;
    int CurrentMip = size.z;
    int count = 1;
    int TotalSteps = 0;
    int face = 5;
    int previous_face = 5;
    cudaTextureObject_t texf = tex.N_mz;

    face = getFaceNumber(r.o, texf, tex);
    float3 uv = world2face(r.o, face);
    //float3 uv_d = world2face(r.d, face);
    float3 uv_d = world2face(r.o + r.d, face) - uv;

    int MinMip = size.z + 1;
    int minFace = 5;
    float3 minuv = make_float3(0.0f);
    for (int i = 0; i < maxIter; ++i) {
        if (CurrentMip == levelOfDetails - 1 || CurrentMip == size.z + 1) break;

        previous_face = face;

        if (uv.x <= -1.0f + eps && uv_d.x < 0.0f) { uv.x -= 2.0f * eps; r.o = face2world(uv, face); previous_face = -1;}
        if (uv.x >= 1.0f - eps && uv_d.x > 0.0f) { uv.x += 2.0f * eps; r.o = face2world(uv, face); previous_face = -1;}
        if (uv.y <= -1.0f + eps && uv_d.y < 0.0f) { uv.y -= 2.0f * eps; r.o = face2world(uv, face); previous_face = -1;}
        if (uv.y >= 1.0f - eps && uv_d.y > 0.0f) { uv.y += 2.0f * eps; r.o = face2world(uv, face); previous_face = -1;}
        if (uv.z >= 1.0f-eps) {
            //перенести луч
            float tnear, tfar;
            const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
            const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);
            intersectBox(r, boxMin, boxMax, tnear, tfar);
            r.o = r.o + tfar * r.d;
            //uv = world2face(r.o, face);
            //uv_d = world2face(r.o + r.d, face) - uv;
            previous_face = -1;
        }
        //previous_face = face;
        //face = getFaceNumber(r.o, texf, tex);
        //if (face != previous_face) { 
        if (previous_face < 0) {
            CurrentMip = size.z;  count = 1;
            face = getFaceNumber(r.o, texf, tex); // added
            uv = world2face(r.o, face);
            uv_d = world2face(r.o + r.d, face) - uv;
        }

        bool above = false;
        float Z;
      
        for (; above == false && CurrentMip >= levelOfDetails; --CurrentMip) {
            float mippow = float(1 << CurrentMip);
            //float x = floor(uv.x / mippow);
            //float y = floor(uv.y / mippow);
            float x = floor((uv.x+1.0f) * size.x * 0.5f / mippow);
            float y = floor((uv.y+1.0f) * size.y * 0.5f / mippow);

            xlim = { x * mippow, (x + 1.0f) * mippow };
            ylim = { y * mippow, (y + 1.0f) * mippow };

            //if (uv.x == xlim.x && uv_d.x < 0) {
            if ((uv.x + 1.0f) * size.x * 0.5f == xlim.x && uv_d.x < 0.0f) {
                xlim -= mippow;
                x -= 1.0f;
            }
            //if (uv.y == ylim.x && uv_d.y < 0) {
            if ((uv.y + 1.0f) * size.y * 0.5f == ylim.x && uv_d.y < 0.0f) {
                ylim -= mippow;
                y -= 1.0f;
            }

            x = fminf(fmaxf(x, 0.0f), (size.x - 1));
            y = fminf(fmaxf(y, 0.0f), (size.y - 1));
            Z = (tex2DLod<float4>(texf, (x + 0.5f) * mippow / size.x, (y + 0.5f) * mippow / size.y, CurrentMip)).w;
            above = (Z < uv.z);
        }

        if (above == false) break;
        else ++CurrentMip;

        float z1 = (-uv.z + Z) / uv_d.z;
        float z2 = (-uv.z + 1.0f) / uv_d.z;
        //float x1 = (-uv.x + xlim.x) / uv_d.x;
        //float x2 = (-uv.x + xlim.y) / uv_d.x;
        //float y1 = (-uv.y + ylim.x) / uv_d.y;
        //float y2 = (-uv.y + ylim.y) / uv_d.y;
        float x1 = (-uv.x + ((xlim.x / size.x) * 2.0f - 1.0f)) / uv_d.x;
        float x2 = (-uv.x + ((xlim.y / size.x) * 2.0f - 1.0f)) / uv_d.x;
        float y1 = (-uv.y + ((ylim.x / size.y) * 2.0f - 1.0f)) / uv_d.y;
        float y2 = (-uv.y + ((ylim.y / size.y) * 2.0f - 1.0f)) / uv_d.y;
        x1 = max(x1, x2);
        y1 = max(y1, y2);
        float z3 = max(max(z1, z2), 0.0f);
        dist = min(min(x1, y1), z3);
        uv = uv + dist * uv_d;
        //r.o = face2world(uv, face);

        // ход вглубь, если луч пересекает Z грань или ниже ее
        if (dist == z1) {
            CurrentMip = CurrentMip - 1;
            count = 1;
            if (MinMip <= CurrentMip) {
                minFace = face;
                minuv = uv;
            }
        }
        else {//% ход в сторону и наружу
            if (count % 3 == 0)
                CurrentMip = CurrentMip + 1;
            count = count + 1;
        }
        TotalSteps = TotalSteps + 1;
    }
    if (CurrentMip == levelOfDetails - 1)
        r.o = face2world(uv, face);
    else
        r.o = face2world(minuv, minFace);
    float3 tmp = (r0 - r.o);
    //return sqrtf(dot(tmp, tmp));//TotalDist;
    //return float(TotalDist);
    //return float(TotalSteps);
    //return float(face);
    return make_float2(sqrtf(dot(tmp, tmp)), (float)TotalSteps);//TotalDist;
}

__device__ inline float4 logf(float4 a) { return make_float4(logf(a.x), logf(a.y), logf(a.z), logf(a.w)); }
__device__ inline float4 clampOne(float4 a) { return make_float4(fmin(a.x,1.0f), fmin(a.y, 1.0f), fmin(a.z, 1.0f), fmin(a.w, 1.0f)); }

__global__ void raymarchkernel(uchar4* d_output, unsigned int imageW, unsigned int imageH, cubetex tex, int3 size, float brightness)
{
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

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    float3 pos = eyeRay.o;// +eyeRay.d * tnear;
    //float3 step = eyeRay.d * tstep;

    //eyeRay.o = pos + /*eps **/ 0.1f * eyeRay.d;
    float dist = MarchRay(eyeRay, tex, size, 256).x;
    if (dist < 0) return;
    //pos = eyeRay.o + (dist + eps) * eyeRay.d;
    pos = eyeRay.o + dist * eyeRay.d;
    //pos = eyeRay.d*sqrt(2.0f);

    // write output color
    //sum = sample(tex, size, eyeRay.d, 0.0f);
    sum = sample(tex, size, pos, 0.0f);
    
    float3 normals = make_float3(sampleNormals(tex, size, pos, 0.0f));
    Ray ReflectionRay;
    ReflectionRay.d = reflect(eyeRay.d, normals);
    //NormalRay.o = pos + 10 * eps * NormalRay.d;
    //ReflectionRay.o = pos * 0.99f;
    ReflectionRay.o = pos + 500.0f * eps * normals;
    __syncthreads();
    float normals_dist =  MarchRay(ReflectionRay, tex, size, 255).x;
    float3 pos2 = ReflectionRay.o + normals_dist * ReflectionRay.d;
    sum = sum + sample(tex, size, pos2, 0.0f);

    //float2 source_dist = intersectSphere(ReflectionRay, make_float3(0.0f), 0.5f);
    //if (source_dist.x >=0.0f && source_dist.x < normals_dist) sum = make_float4(10.f);

    sum = logf(sum*3.0f+1.0f);
    //sum = clampOne(sum);
    sum = clamp(sum, 0.0f, 1.f);
    sum.w = 1.0f;
    sum *= 255.0f;
    //dist *= 128.0f;
    //sum = make_float4(dist, dist, dist,255.0f);
    //sum = make_float4(normals_dist, normals_dist, normals_dist, 255.0f);
    sum = clamp(sum, 0.0f, 255.0f);
    d_output[y * imageW + x] = to_uchar4(sum);
}

//-----------------------------------------------------------------------------------------------------

//__constant__  int iters[] = {256,128,64,32,32,32,32,16};
//__constant__  int details[] = { 0,0,1,1,2,3,3,4 };

__device__ /*float4*/void MarchRayM(Ray r, cubetex tex, int3 size, float4* rays_info, int maxIter = 128, int maxReflections = 2, int levelOfDetails = 0) {
    //int iters[] = {256,128,64,32,32,32,32,16};
    //int details[] = { 0,0,1,1,2,3,3,4 };
    for (int i = 0; i < maxReflections; ++i) {
        //float dist = MarchRay(r, tex, size, maxIter, levelOfDetails).x;
        //float dist = MarchRay(r, tex, size, (1 << (8 - i / 2)), i / 2).x;
        //loat dist = MarchRay(r, tex, size, (int)(256 / (i+1)), i).x;
        //int iters = maxIter;// __float2int_rd(1 << (8 - __float2int_rd(i / 2)));
        //int details = levelOfDetails;// __float2int_rd((float)i * 0.5f);
        //if (i == 0) { iters = maxIter; details = levelOfDetails; }
        ////else if (i < 2) { iters = 128; details = 1; }
        //else if (i < 3) { iters = 64; details = 1; }
        //else { iters = 32; details = 2; }
        //float dist = MarchRay(r, tex, size, maxIter, levelOfDetails).x;
        float dist = MarchRay(r, tex, size, (1 << (8 - i / 2)), i / 2).x;
        r.o = r.o + dist * r.d;
        float4 col = sample(tex, size, r.o, levelOfDetails);
        float4 data = sampleNormals(tex, size, r.o, levelOfDetails);
        float3 normals = make_float3(data);
        rays_info[i] = col;
        r.d = reflect(r.d, normals);
        r.o = r.o * 0.99f;
        if (col.w < eps) break;
        //r.o = r.o - normalize(r.o) * 2.0f * eps;
        //r.o = r.o + 500.0f * eps * normals;
        
    }
    return;
}

__device__ float4 ComputeColor(float4* rays_info, int maxReflections = 2) {
    //float4 color = make_float4(0.0f);
    float4 color = rays_info[maxReflections - 1];
    for (int i = maxReflections - 2; i >= 0; --i) {
        float4 data = rays_info[i];
        color = data * (1.0f - data.w) + data.w * color;
    }
    return color;
}

__global__ void raymarchkernelMultipleReflections(uchar4* d_output, unsigned int imageW, unsigned int imageH, cubetex tex, int3 size, float brightness)
{
    //extern __shared__ float4 s[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //float4* rays_info = s + (blockDim.x * threadIdx.y + threadIdx.x) * MAX_REFL;
    //float4* rays_info = (float4*)malloc(MAX_REFL*sizeof(float4));
    float4 rays_info[MAX_REFL];// = (float4*)malloc(MAX_REFL * sizeof(float4));
    for (int i = 0; i < MAX_REFL; ++i) rays_info[i] = make_float4(0.0f);

    if ((x >= imageW) || (y >= imageH)) return;

    float ratio = (float)imageW / imageH;
    //float u = (x * / (float)imageW) * 2.0f - 1.0f;
    float u = ratio * ((x / (float)imageW) * 2.0f - 1.0f);
    float v = (y / (float)imageH) * 2.0f - 1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // march along ray from front to back, accumulating color
    float4 sum;// = make_float4(0.0f);
    //float3 pos = eyeRay.o;

    const int maxReflections = 8;
    const int levelOfDetails = 0;
    /*sum =*/ MarchRayM(eyeRay, tex, size, rays_info, 256, maxReflections, levelOfDetails);
    //__syncthreads();
    sum = ComputeColor(rays_info, maxReflections);
    //sum = sampleNormals(tex, size, eyeRay.d, 5);
    //float3 normals = make_float3(data);
    //float reflectivity = data.w;
    //sum = make_float4(sum.w)/1.0f;

    sum *= brightness;
    sum = logf(sum * 3.0f + 1.0f);
    //sum = clampOne(sum);
    sum = clamp(sum, 0.0f, 1.f);
    sum.w = 1.0f;
    sum *= 255.0f;
    //dist *= 128.0f;
    //sum = make_float4(dist, dist, dist,255.0f);
    //sum = make_float4(normals_dist, normals_dist, normals_dist, 255.0f);
    sum = clamp(sum, 0.0f, 255.0f);
    //free(rays_info);
    d_output[y * imageW + x] = to_uchar4(sum);
}

__device__ float MarchRaySound(Ray r, cubetex tex, int3 size, float4& rayinfo, int maxIter = 128, int maxReflections = 2, int levelOfDetails = 0, float alpha = 0.005f, float3 source_pos = { 0.0f,0.0f,0.0f }, float radius = 0.5f) {//, int cheapReflections = 5) {
    rayinfo = make_float4(1.0f,1.0f,1.0f,1.0f);
    float TotalDistance = 0.0f;
    for (int i = 0; i < maxReflections; ++i) {
        //int iters = maxIter;
        //int details = levelOfDetails;
        //if (i == 0) { iters = maxIter; details = levelOfDetails; }
        //else if (i < 2) { iters = 128; details = 1; }
        //else if (i < 3) { iters = 64; details = 1; }
        //else { iters = 32; details = 2; }
        float ray_width = /*2.0f */ alpha * TotalDistance;
        float2 source_dist = intersectSphere(r, source_pos, radius + ray_width);
        float dist = MarchRay(r, tex, size, (1 << (8 - i / 2)), i / 2).x;
        r.o = r.o + dist * r.d;
        float4 col = sample(tex, size, r.o, levelOfDetails);
        float4 data = sampleNormals(tex, size, r.o, levelOfDetails);
        float3 normals = make_float3(data);
        r.d = reflect(r.d, normals);
        r.o = r.o * 0.99f;
        if (rayinfo.x + rayinfo.y + rayinfo.z + rayinfo.w < 0.00001f) { return -1.0f; }
        //r.o = r.o - normalize(r.o) * 2.0f * eps;
        //r.o = r.o + 500.0f * eps * normals;
        if (source_dist.x >= 0.0f && source_dist.x < dist) { 
            TotalDistance += source_dist.x;
            return TotalDistance;
        };
        //rayinfo *= col;
        TotalDistance += dist;
    }
    /*
    for (int i = 0; i < cheapReflections; ++i) {
        //float ray_width = alpha * TotalDistance;
        float2 source_dist = intersectSphere(r, make_float3(0.0f), 0.5f );
        
        //float4 point = sampleNormals(tex, size, r.o, levelOfDetails);
        float depth = 0.0f;
        float4 s = sampleNormalsAndDepth(tex, size, r.d, levelOfDetails, depth);
        float3 point_next = normalize(r.d) * depth;
        float dist = length(point_next - r.o);

        r.o = point_next;
        float4 col = sample(tex, size, r.o, levelOfDetails);
        float3 normals = make_float3(s);
        r.d = reflect(r.d, normals);
        r.o = r.o * 0.99f;
        if (rayinfo.x + rayinfo.y + rayinfo.z + rayinfo.w < 0.00001f) { return -1.0f; }
        //r.o = r.o - normalize(r.o) * 2.0f * eps;
        //r.o = r.o + 500.0f * eps * normals;
        if (source_dist.x >= 0.0f && source_dist.x < dist) {
            TotalDistance += source_dist.x;
            return TotalDistance;
        };
        //rayinfo *= col;
        TotalDistance += dist;
    }*/
    return -1.0f;
}

__global__ void raymarchkernelSound(uchar4* d_output, unsigned int imageW, unsigned int imageH, cubetex tex, int3 size, float4* IR, float brightness)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    //float4 rays_info[MAX_REFL];// = (float4*)malloc(MAX_REFL * sizeof(float4));
    //for (int i = 0; i < MAX_REFL; ++i) rays_info[i] = make_float4(0.0f);

    if ((x >= imageW) || (y >= imageH)) return;

    float ratio = (float)imageW / imageH;
    //float u = (x * / (float)imageW) * 2.0f - 1.0f;
    float u = ratio * ((x / (float)imageW) * 2.0f - 1.0f);
    float v = (y / (float)imageH) * 2.0f - 1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    //float3 pos = eyeRay.o;
    float4 rayinfo;// = make_float4(0.0f); */

    const int maxReflections = 8;
    const int levelOfDetails = 1;
    /*sum =*/
    float alpha = 2.0f * 2.0f / sqrtf((float)imageW * (float)imageH);
    float distance = MarchRaySound(eyeRay, tex, size, rayinfo, 128, maxReflections, levelOfDetails, alpha);
    sum = rayinfo; 
    if (distance <= 0.0f) { sum = make_float4(0.0f);}
    float4 res = sum;
    float4 res1 = make_float4(0.0f, u, v, distance);
    IR[(y * imageW + x)*2] = res;
    IR[(y * imageW + x) * 2 + 1] = res1;

    sum *= brightness;
    sum = logf(sum * 3.0f + 1.0f);
    //sum = clampOne(sum);
    sum = clamp(sum, 0.0f, 1.f);
    sum.w = 1.0f;
    sum *= 255.0f;
    //dist *= 128.0f;
    //sum = make_float4(dist, dist, dist,255.0f);
    //sum = make_float4(normals_dist, normals_dist, normals_dist, 255.0f);
    //sum = clamp(sum, 0.0f, 255.0f);
    //free(rays_info);
    d_output[y * imageW + x] = to_uchar4(sum);
}

__global__ void raymarchkernelSoundCube(uchar4* d_output, unsigned int side_N, unsigned int imageW, unsigned int imageH, cubetex tex, int3 size, float4* IR, float brightness)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int face = blockIdx.z;

    if ((x >= side_N) || (y >= side_N)) return;

    float u = (x / (float)side_N) * 2.0f - 1.0f;
    float v = (y / (float)side_N) * 2.0f - 1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(face2world(make_float3(u,v,1.0f),face));//normalize(make_float3(u, v, -2.0f));
    //eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);
    //float3 pos = eyeRay.o;
    float4 rayinfo;// = make_float4(0.0f); */

    const int maxReflections = 8;
    const int levelOfDetails = 1;
    /*sum =*/
    float alpha = 2.0f * 2.0f / (sqrtf(6.0f) * side_N);// sqrtf((float)imageW * (float)imageH);
    float distance = MarchRaySound(eyeRay, tex, size, rayinfo, 128, maxReflections, levelOfDetails, alpha);
    sum = rayinfo;
    if (distance <= 0.0f) { sum = make_float4(0.0f); }
    float4 res = sum;
    float4 res1 = make_float4(0.0f, u, v, distance);
    IR[(face * side_N * side_N + y * side_N + x) * 2] = res;
    IR[(face * side_N * side_N + y * side_N + x) * 2 + 1] = res1;

    sum *= brightness;
    sum = logf(sum * 3.0f + 1.0f);
    //sum = clampOne(sum);
    sum = clamp(sum, 0.0f, 1.f);
    sum.w = 1.0f;
    sum *= 255.0f;

    int xi = x + (side_N * (face % 3));
    int yi = y + (side_N * (face / 3));
    if (xi >= imageW || yi >= imageH) return;
    d_output[yi * imageW + xi] = to_uchar4(sum);
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
        color /= 4.0f;
        
        if (modeMax)
            color.w = max(max((tex2D<float4>(mipInput, (x + 0) * px, (y + 0) * py)).w,
            (tex2D<float4>(mipInput, (x + 1) * px, (y + 0) * py)).w),
            max((tex2D<float4>(mipInput, (x + 1) * px, (y + 1) * py)).w,
            (tex2D<float4>(mipInput, (x + 0) * px, (y + 1) * py)).w));
        
        //color *= 255.0;
        //color = fminf(color, make_float4(255.0));

        //surf2Dwrite(to_uchar4(color), mipOutput, x * sizeof(uchar4), y);
        surf2Dwrite(color, mipOutput, x * sizeof(float4), y);
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
        texDescr.filterMode = cudaFilterModePoint;
        //texDescr.filterMode = cudaFilterModeLinear;

        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;
        //texDescr.addressMode[0] = cudaAddressModeWrap;
        //texDescr.addressMode[1] = cudaAddressModeWrap;
        //texDescr.addressMode[2] = cudaAddressModeWrap;

        //texDescr.readMode = cudaReadModeNormalizedFloat;
        texDescr.readMode = cudaReadModeElementType;

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
void initCuda(std::vector<Image> &images)
{
    scene.resize(12);
    for (int i = 0; i < 12; ++i) {
        Image& image = scene[i];
        image.size = images[i].size;
        image.size.depth = 0;
        image.type = cudaResourceTypeMipmappedArray;
        // how many mipmaps we need
        uint levels = getMipMapLevels(image.size);
        highestLod = MAX(highestLod, (float)levels - 1);

        // create 2D array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        cudaArray_t level0;

        checkCudaErrors(cudaMallocMipmappedArray(&image.mipmapArray, &channelDesc, image.size, levels));
        checkCudaErrors(cudaGetMipmappedArrayLevel(&level0, image.mipmapArray, 0));

        // загрузка в память нулевого мипа
        checkCudaErrors(cudaMemcpyToArray(level0, 0, 0, images[i].h_data, image.size.width * image.size.height * sizeof(float4), cudaMemcpyHostToDevice));

        // compute rest of mipmaps based on level 0
        generateMipMaps(image.mipmapArray, image.size, (i>=6));

        // generate bindless texture object

        cudaResourceDesc            resDescr;
        memset(&resDescr, 0, sizeof(cudaResourceDesc));

        resDescr.resType = cudaResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap = image.mipmapArray;

        cudaTextureDesc             texDescr;
        memset(&texDescr, 0, sizeof(cudaTextureDesc));

        texDescr.normalizedCoords = 1;
        //if (i < 6) {        
        //    texDescr.filterMode = cudaFilterModePoint;
        //    texDescr.mipmapFilterMode = cudaFilterModePoint;
        //}
        //else {
            texDescr.filterMode = cudaFilterModeLinear;
            texDescr.mipmapFilterMode = cudaFilterModeLinear;
        //}


        texDescr.addressMode[0] = cudaAddressModeClamp;
        texDescr.addressMode[1] = cudaAddressModeClamp;
        texDescr.addressMode[2] = cudaAddressModeClamp;
        //texDescr.addressMode[0] = cudaAddressModeWrap;
        //texDescr.addressMode[1] = cudaAddressModeWrap;
        //texDescr.addressMode[2] = cudaAddressModeWrap;

        texDescr.maxMipmapLevelClamp = float(levels - 1);

        //texDescr.readMode = cudaReadModeNormalizedFloat;
        texDescr.readMode = cudaReadModeElementType;

        checkCudaErrors(cudaCreateTextureObject(&image.textureObject, &resDescr, &texDescr, NULL));

        image.size.depth = (int)highestLod;
    }
    sceneCube.px = scene[0].textureObject;
    sceneCube.mx = scene[1].textureObject;
    sceneCube.py = scene[2].textureObject;
    sceneCube.my = scene[3].textureObject;
    sceneCube.pz = scene[4].textureObject;
    sceneCube.mz = scene[5].textureObject;
    sceneCube.N_px = scene[6].textureObject;
    sceneCube.N_mx = scene[7].textureObject;
    sceneCube.N_py = scene[8].textureObject;
    sceneCube.N_my = scene[9].textureObject;
    sceneCube.N_pz = scene[10].textureObject;
    sceneCube.N_mz = scene[11].textureObject;
}