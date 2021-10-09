#pragma once
#include "CudaModulePCH.h"
#include "Julia.h"
#include "core/CudaContext.h"
#include "utility/CpuBitmap.h"
#include "GraphicInterop/CudaTexture.h"

struct cuComplex {
    float   r;
    float   i;
    __device__ cuComplex(float a, float b) : r(a), i(b) {}
    __device__ float magnitude2(void) {
        return r * r + i * i;
    }
    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

__device__ float julia(int x, int y) {
    const float scale = .08;
    float jx = scale * ((float)(DIM / 2 - x) / (DIM / 2) - 17);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
   

    cuComplex a(0, 0);
    cuComplex c(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 500)
            return i * 1.0 / 100;
    }

    return 0;
}

__global__ void kernel(unsigned char* ptr, cudaTextureObject_t lut) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    const float scale = 1;
    float jx = scale * (float)(x) / (DIM);
    float jy = scale * (float)(y) / (DIM);

    // now calculate the value at that position
    float juliaValue = julia(x, y);
    uint4 color = tex2D<uint4>(lut, juliaValue, 0.5);
    ptr[offset * 4 + 0] = color.x;
    ptr[offset * 4 + 1] = color.y;
    ptr[offset * 4 + 2] = color.z;
    ptr[offset * 4 + 3] = 255;
}

// globals needed by the update routine
struct DataBlock {
    unsigned char* dev_bitmap;
};

void SIByL::CudaContext::Julia()
{
    //DataBlock   data;
    //CPUBitmap bitmap(DIM, DIM, &data);
    //unsigned char* dev_bitmap;

    //CudaTexture lut("../Assets/Resources/Textures/lut.png");
    //cudaTextureObject_t textureObject = lut.GetTextureObject();

    //HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
    //data.dev_bitmap = dev_bitmap;

    //dim3    grid(DIM, DIM);
    //kernel <<<grid, 1 >>> (dev_bitmap, textureObject);

    //HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
    //    bitmap.image_size(),
    //    cudaMemcpyDeviceToHost));

    //HANDLE_ERROR(cudaFree(dev_bitmap));

    //bitmap.display_and_exit();
}