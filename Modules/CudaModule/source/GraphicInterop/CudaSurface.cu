#pragma once

#include "CudaModulePCH.h"
#include "CudaSurface.h"
#include "Sibyl/Graphic/Core/Texture/Image.h"

namespace SIByL
{
	CudaSurface::CudaSurface(const std::string& path)
	{
		Image image(path);

		const int width = image.GetWidth();
		const int height = image.GetHeight();
		unsigned char* data = image.GetData();

		// allocate cuda array in device memory
		cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaArray* cuArray;
		cudaMallocArray(&cuArray, &channelDesc, width, height, cudaArraySurfaceLoadStore);

		// Set pitch of the source (the width in memory in bytes of the 2D array pointed
		// to by src, including padding), we dont have any padding
		size_t bytesPerElem = sizeof(uchar4);
		// Copy data located at address h_data in host memory to device memory
		checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, data, width * bytesPerElem,
			width * bytesPerElem, height, cudaMemcpyHostToDevice));

		// specify texture
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// !FAILS! create texture object
		checkCudaErrors(cudaCreateSurfaceObject(&mSurfaceObject, &resDesc));
	}

	cudaSurfaceObject_t CudaSurface::GetSurfaceObject()
	{
		return mSurfaceObject;
	}

	void CudaSurface::RegisterByOpenGLTexture(uint32_t id, uint32_t width, uint32_t height)
	{
		Width = width;
		Height = height;
		cudaGraphicsUnregisterResource(cudaBuffer);
		checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaBuffer, id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
	}

	void CudaSurface::StartOpenGLMapping()
	{
		checkCudaErrors(cudaGraphicsMapResources(1, &cudaBuffer, 0));

		cudaArray_t cuArray = nullptr;
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&cuArray, cudaBuffer, 0, 0));

		// specify texture
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		checkCudaErrors(cudaCreateSurfaceObject(&mSurfaceObject, &resDesc));
	}

	void CudaSurface::EndOpenGLMapping()
	{
		checkCudaErrors(cudaDestroySurfaceObject(mSurfaceObject));
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaBuffer, 0));
		checkCudaErrors(cudaStreamSynchronize(0));
	}
}