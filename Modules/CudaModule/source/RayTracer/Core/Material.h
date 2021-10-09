#pragma once

#include <CudaModulePCH.h>

class CUDARay;
class CUDAHitRecord;

class CUDAMaterial
{
public:
	__device__ virtual ~CUDAMaterial() {}
	__device__ virtual bool scatter(
		const CUDARay& r_in,
		const CUDAHitRecord& rec, 
		float3& attenuation,
		float3& emission, 
		CUDARay& scattered) const = 0;
};