#pragma once

#include "CudaModulePCH.h"

class CUDARay;
class CUDAMaterial;
class CUDAHitRecord;

class CUDAHitable
{
public:
    __device__ CUDAHitable(CUDAMaterial* mat) {}
    __device__ virtual ~CUDAHitable() {}
    __device__ virtual bool hit(const CUDARay& r, float t_min, float t_max, CUDAHitRecord& rec) const = 0;
};