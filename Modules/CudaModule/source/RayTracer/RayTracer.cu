#pragma once

#include "CudaModulePCH.h"
#include "RayTracer.h"
#include "surface_functions.h"
#include "GraphicInterop/CudaSurface.h"
#include "RayTracer/Core/Primitive.h"


#define INF 2e10f  
#define rnd(x) (x*rand()/RAND_MAX)  
#define SPHERES 100 //球体数量

struct Sphere
{
    float r, g, b;
    float radius;
    float x, y, z;

    __device__ float hit(float ox, float oy, float* n)
    {
        float dx = ox - x;
        float dy = oy - y;

        if (dx * dx + dy * dy < radius * radius)
        {
            float dz = sqrt(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrt(radius * radius);
            return dz + z;
        }

        return -INF;
    }
};

// Sphere *s;
__constant__ Sphere s[SPHERES];

__global__ void RayTracerKernel(cudaSurfaceObject_t surface, int width, int height) {
    // map from blockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    // if the pixel is out of the surface size
    if (x >= width || y >= height) return;

    float u = 1.0 * x / width;
    float v = 1.0 * y / height;

    float ox = (x - width / 2);
    float oy = (y - height / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++)
    {
        float n;
        float t = s[i].hit(ox, oy, &n);
        if (t > maxz)
        {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    uchar4 color = make_uchar4(r * 255, g * 255, b * 255, 255);
    surf2Dwrite(color, surface, x * 4, y);
    //int offset = x + y * gridDim.x;

    //const float scale = 1;
    //float jx = scale * (float)(x) / (DIM);
    //float jy = scale * (float)(y) / (DIM);

    //// now calculate the value at that position
    //float juliaValue = julia(x, y);
    //uint4 color = tex2D<uint4>(lut, juliaValue, 0.5);
    //ptr[offset * 4 + 0] = color.x;
    //ptr[offset * 4 + 1] = color.y;
    //ptr[offset * 4 + 2] = color.z;
    //ptr[offset * 4 + 3] = 255;
}

Sphere* temps = nullptr;
float* velocities = nullptr;

namespace SIByL
{
    void RayTracer::RenderPtrCudaSurface(CudaSurface* pCudaSurface, float deltaTime)
    {
        if (temps == nullptr)
        {
            delete[] temps;
            temps = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
            velocities = (float*)malloc(sizeof(float) * SPHERES);
            srand(time(0));  //随机数种子

            for (int i = 0; i < SPHERES; i++)
            {
                velocities[i] = 0;
                temps[i].r = rnd(1.0f);
                temps[i].g = rnd(1.0f);
                temps[i].b = rnd(1.0f);
                temps[i].x = rnd(1000) - 500;
                temps[i].y = rnd(600) - 300;
                temps[i].z = rnd(1000) - 500;
                temps[i].radius = rnd(30.0f) + 20;
            }

            //  cudaMemcpy(s, temps, sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice);  
            //cudaMemcpyToSymbol(s, temps, sizeof(Sphere) * SPHERES);
            //free(temps);
        }

        for (int i = 0; i < SPHERES; i++)
        {
            velocities[i] -= 9.8 * deltaTime;
            velocities[i] *= 0.999999;
            temps[i].y += velocities[i];
            if (temps[i].y < -300) velocities[i] = -velocities[i];
        }

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        cudaMemcpyToSymbol(s, temps, sizeof(Sphere) * SPHERES);

        uint32_t width = pCudaSurface->GetWidth();
        uint32_t height = pCudaSurface->GetHeight();
        int thread_size = 16;
        dim3 grids(ceil(1.0 * width / thread_size), ceil(1.0 * height / thread_size));
        dim3 threads(thread_size, thread_size);
        RayTracerKernel << <grids, threads >> > (pCudaSurface->GetSurfaceObject(), width, height);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << elapsedTime << std::endl;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}