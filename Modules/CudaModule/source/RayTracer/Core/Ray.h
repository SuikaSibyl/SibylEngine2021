#pragma once

#include "CudaModulePCH.h"
#include "helper_math.h"
#include "Material.h"

class CUDARay
{
public:
    __device__ CUDARay() :
        origin(make_float3(0, 0, 0)), direction(make_float3(0, 0, 0)) {}

    __device__ CUDARay(float3 iorigin, float3 idir) :
        origin(iorigin), direction(idir) {}

    __device__ float3 pointAtParameter(float t) const
    {
        return origin + direction * t;
    }

    float3 origin;
    float3 direction;
};

struct CUDAHitRecord
{
    float t;
    float3 p;
    float3 normal;
    CUDAMaterial* material;
};
