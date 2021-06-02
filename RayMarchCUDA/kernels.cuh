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

extern "C" void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix);
extern "C" void freeCudaBuffers();
extern "C" void initCuda(Image& image_in);
extern "C" void raymarchkernelStart(uchar4 * d_output, unsigned int imageW, unsigned int imageH, dim3 grid, dim3 block);
__global__ void raymarchkernel(uchar4* d_output, unsigned int imageW, unsigned int imageH, cudaTextureObject_t tex, int3 size, float height);// = NULL);