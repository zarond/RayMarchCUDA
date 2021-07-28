// includes, cuda
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_math.h>
#include <vector_types.h>


__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
    return make_uchar4((unsigned char)vec.x, (unsigned char)vec.y, (unsigned char)vec.z, (unsigned char)vec.w);
}

typedef unsigned int  uint;
#pragma pack(push,4)
struct Image
{
    void* h_data;
    cudaExtent              size;
    cudaResourceType        type;
    cudaArray_t             dataArray;
    cudaMipmappedArray_t    mipmapArray;
    cudaTextureObject_t     textureObject;

    Image()
    {
        memset(this, 0, sizeof(Image));
    }
};
#pragma pack(pop)

typedef struct __device__ __host__ cubetex {
    cudaTextureObject_t  px;
    cudaTextureObject_t  mx;
    cudaTextureObject_t  py;
    cudaTextureObject_t  my;
    cudaTextureObject_t  pz;
    cudaTextureObject_t  mz;
    cudaTextureObject_t  N_px;
    cudaTextureObject_t  N_mx;
    cudaTextureObject_t  N_py;
    cudaTextureObject_t  N_my;
    cudaTextureObject_t  N_pz;
    cudaTextureObject_t  N_mz;
};

#pragma pack(push,4)
struct ImageCube
{
    void* h_data[6] = {0,0,0,0,0,0};
    cudaExtent              size;
    cudaResourceType        type;
    cudaArray_t             dataArray;
    cudaMipmappedArray_t    mipmapArray[6];
    //cudaTextureObject_t     textureObject[6];
    //cubemiparr mipmapArray;
    cubetex texs;

    ImageCube()
    {
        memset(this, 0, sizeof(ImageCube));
    }
};
#pragma pack(pop)

extern "C" void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix);
//extern "C" void freeCudaBuffers();
extern "C" void freeCudaBuffers();
extern "C" void initCuda(std::vector<Image> &images);
extern "C" void raymarchkernelStart(uchar4 * d_output, unsigned int imageW, unsigned int imageH, dim3 grid, dim3 block, float4 * IR = nullptr, int intensitySetting=0, int mode=1);
__global__ void raymarchkernel(uchar4* d_output, unsigned int imageW, unsigned int imageH, cubetex tex, int3 size, float brightness = 3.0f);
__global__ void raymarchkernelMultipleReflections(uchar4* d_output, unsigned int imageW, unsigned int imageH, cubetex tex, int3 size, float brightness = 3.0f);
__global__ void raymarchkernelSound(uchar4* d_output, unsigned int imageW, unsigned int imageH, cubetex tex, int3 size, float4* IR, float brightness = 1.0f);
__global__ void raymarchkernelSoundCube(uchar4* d_output, unsigned int side_N, unsigned int imageW, unsigned int imageH, cubetex tex, int3 size, float4* IR, float brightness = 1.0f);