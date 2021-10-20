#pragma once

#include "CudaModulePCH.h"
#include "CuBVH.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

namespace SIByL
{
	typedef unsigned int MortonType;

	struct MortonRec {
		float x, y, z;
		float xx, yy, zz;
		MortonType ex, ey, ez;
		MortonType m;
	};

	void CuBVH::LoadData(const std::vector<float>& vertices, const std::vector<uint32_t>& indices, unsigned int stepsize)
	{
		std::vector<float> triangles;
		// step 9: v1_x, v1_y, v1_z, v2_x, v2_y, v2_z, v3_x, v3_y, v3_z
		std::vector<float> boundingboxes;
		// step 6: xmin, xmax. ymin, ymax, zmin, zmax

		for (unsigned int i = 0; i < indices.size(); i += 3)
		{
			float v1x = vertices[stepsize * indices[i + 0] + 0];
			float v1y = vertices[stepsize * indices[i + 0] + 1];
			float v1z = vertices[stepsize * indices[i + 0] + 2];

			float v2x = vertices[stepsize * indices[i + 1] + 0];
			float v2y = vertices[stepsize * indices[i + 1] + 1];
			float v2z = vertices[stepsize * indices[i + 1] + 2];

			float v3x = vertices[stepsize * indices[i + 2] + 0];
			float v3y = vertices[stepsize * indices[i + 2] + 1];
			float v3z = vertices[stepsize * indices[i + 2] + 2];

			triangles.push_back(v1x);
			triangles.push_back(v1y);
			triangles.push_back(v1z);
			triangles.push_back(v2x);
			triangles.push_back(v2y);
			triangles.push_back(v2z);
			triangles.push_back(v3x);
			triangles.push_back(v3y);
			triangles.push_back(v3z);

			boundingboxes.push_back(fmin(fmin(v1x, v2x), v3x));
			boundingboxes.push_back(fmax(fmax(v1x, v2x), v3x));
			boundingboxes.push_back(fmin(fmin(v1y, v2y), v3y));
			boundingboxes.push_back(fmax(fmax(v1y, v2y), v3y));
			boundingboxes.push_back(fmin(fmin(v1z, v2z), v3z));
			boundingboxes.push_back(fmax(fmax(v1z, v2z), v3z));

		}
		// must be bounded to unit cube
		float bounds[6] = { FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX, FLT_MAX, -FLT_MAX };
		for (int i = 0; i < boundingboxes.size(); i += 6)
		{
			bounds[0] = fmin(bounds[0], boundingboxes[i + 0]);
			bounds[1] = fmax(bounds[1], boundingboxes[i + 1]);
			bounds[2] = fmin(bounds[2], boundingboxes[i + 2]);
			bounds[3] = fmax(bounds[3], boundingboxes[i + 3]);
			bounds[4] = fmin(bounds[4], boundingboxes[i + 4]);
			bounds[5] = fmax(bounds[5], boundingboxes[i + 5]);
		}

		float _scale = fmin(fmin(1.0f / (bounds[1] - bounds[0]), 1.0f / (bounds[3] - bounds[2])), 1.0f / (bounds[5] - bounds[4]));
		for (int i = 0; i < boundingboxes.size(); i += 6)
		{
			boundingboxes[i + 0] = fmax(0.01, fmin(0.99, (boundingboxes[i + 0] - bounds[0]) * _scale));
			boundingboxes[i + 1] = fmax(0.01, fmin(0.99, (boundingboxes[i + 1] - bounds[0]) * _scale));
			boundingboxes[i + 2] = fmax(0.01, fmin(0.99, (boundingboxes[i + 2] - bounds[2]) * _scale));
			boundingboxes[i + 3] = fmax(0.01, fmin(0.99, (boundingboxes[i + 3] - bounds[2]) * _scale));
			boundingboxes[i + 4] = fmax(0.01, fmin(0.99, (boundingboxes[i + 4] - bounds[4]) * _scale));
			boundingboxes[i + 5] = fmax(0.01, fmin(0.99, (boundingboxes[i + 5] - bounds[4]) * _scale));
		}

		BuildBVH(triangles, boundingboxes);
	}

	__device__ MortonType expandBits(MortonType v)
	{
		v = (v * 0x00010001u) & 0xFF0000FFu;
		v = (v * 0x00000101u) & 0x0F00F00Fu;
		v = (v * 0x00000011u) & 0xC30C30C3u;
		v = (v * 0x00000005u) & 0x49249249u;

		return v;
	}

	__global__ void Morton3DKernel(unsigned int triangleCount, MortonType* keys, const float* boundingboxes, MortonRec* records)
	{
		int idx = threadIdx.x + blockDim.x * blockIdx.x;
		if (idx >= triangleCount) return;

		float x, y, z;
		x = (boundingboxes[6 * idx + 0] + boundingboxes[6 * idx + 1]) / 2;
		y = (boundingboxes[6 * idx + 2] + boundingboxes[6 * idx + 3]) / 2;
		z = (boundingboxes[6 * idx + 4] + boundingboxes[6 * idx + 5]) / 2;

		records[idx].x = x;
		records[idx].y = y;
		records[idx].z = z;

		x = x * 1023.0f;
		y = y * 1023.0f;
		z = z * 1023.0f;

		records[idx].xx = x;
		records[idx].yy = y;
		records[idx].zz = z;
		
		MortonType xx = expandBits((MortonType)(x));
		MortonType yy = expandBits((MortonType)(y));
		MortonType zz = expandBits((MortonType)(z));
		keys[idx] = xx * 4 + yy * 2 + zz;

		records[idx].ex = xx;
		records[idx].ey = yy;
		records[idx].ez = zz;
		records[idx].m = keys[idx];
	}

#define THREADS_PER_BLOCK 256

	void CuBVH::BuildBVH(const std::vector<float>& triangles, const std::vector<float>& bbs)
	{
		unsigned int triangleCount = triangles.size() / 9;

		float* d_triangles;
		float* d_boundingboxes;
		MortonType* d_keys;
		MortonRec* d_records;

		checkCudaErrors(cudaMalloc((void**)&d_triangles, sizeof(float) * triangles.size()));
		checkCudaErrors(cudaMalloc((void**)&d_boundingboxes, sizeof(float) * bbs.size()));
		checkCudaErrors(cudaMalloc((void**)&d_keys, sizeof(MortonType) * triangleCount));
		checkCudaErrors(cudaMallocManaged((void**)&d_records, sizeof(MortonRec) * triangleCount));

		checkCudaErrors(cudaMemcpy(d_triangles, triangles.data(), sizeof(float) * triangles.size(), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_boundingboxes, bbs.data(), sizeof(float) * bbs.size(), cudaMemcpyHostToDevice));


		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaDeviceSynchronize();
		cudaEventRecord(start);

		Morton3DKernel<<<(triangleCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
			(triangleCount, d_keys, d_boundingboxes, d_records);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("It took me %f milliseconds to generate morton codes.\n", milliseconds);

		//wrap raw pointer with a device_ptr to use with Thrust functions
		thrust::device_ptr<MortonType> dev_keys_ptr(d_keys);
		thrust::device_ptr<MortonRec>  dev_data_ptr(d_records);

		thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + triangleCount, dev_data_ptr);

		d_records = thrust::raw_pointer_cast(dev_data_ptr);

	}
}