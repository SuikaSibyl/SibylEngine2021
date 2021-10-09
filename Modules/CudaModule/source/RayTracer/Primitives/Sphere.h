#pragma once

#include "RayTracer/Core/Primitive.h"

class CUDARay;
class CUDAMaterial;
class CUDAHitRecord;

class CUDASphere :public CUDAHitable
{
public:
    __device__ CUDASphere(float3 c, float r, CUDAMaterial* mat);
    __device__ virtual ~CUDASphere() {}
    __device__ virtual bool hit(const CUDARay& r, float t_min, float t_max, CUDAHitRecord& rec) const override;

    float3 center;
    float radius;
};