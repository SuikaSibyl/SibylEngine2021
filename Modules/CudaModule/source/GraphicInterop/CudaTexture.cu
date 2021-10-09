#pragma once

#include "CudaModulePCH.h"
#include "CudaTexture.h"
#include "Sibyl/Graphic/Core/Texture/Image.h"

namespace SIByL
{
	CudaTexture::CudaTexture(const std::string& path)
	{
		Image image(path);

		const int width = image.GetWidth();
		const int height = image.GetHeight();
		unsigned char* data = image.GetData();

		// allocate cuda array in device memory
		cudaChannelFormatDesc channelDesc =
			cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaArray* cuArray;
		cudaMallocArray(&cuArray, &channelDesc, width, height);

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

		// specify texture object parameters
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		// !FAILS! create texture object
		checkCudaErrors(cudaCreateTextureObject(&mTextureObject, &resDesc, &texDesc, NULL));
	}

	cudaTextureObject_t CudaTexture::GetTextureObject()
	{
		return mTextureObject;
	}

	CudaTexture* CudaTexture::CreateFromOpenGLTexture(uint32_t id, uint32_t width, uint32_t height)
	{
		cudaGraphicsResource_t  cudaBuffer;

		checkCudaErrors(cudaGraphicsGLRegisterImage(&cudaBuffer, id, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
		checkCudaErrors(cudaGraphicsMapResources(1, &cudaBuffer, 0));

		cudaArray_t* cuArray = nullptr;
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(cuArray, cudaBuffer, 0, 0));

		cudaChannelFormatDesc cuDesc = cudaCreateChannelDesc<uchar4>();
		cudaMallocArray(cuArray, &cuDesc, width, height);

		//checkCudaErrors(cudaMemcpyToArray(cuArray, 0, 0, pResult, width * height * sizeof(uchar4), cudaMemcpyDeviceToDevice));

		//// Set pitch of the source (the width in memory in bytes of the 2D array pointed
		//// to by src, including padding), we dont have any padding
		//size_t bytesPerElem = sizeof(uchar4);
		//// Copy data located at address h_data in host memory to device memory
		//checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, data, width * bytesPerElem,
		//	width * bytesPerElem, height, cudaMemcpyHostToDevice));

		cudaGraphicsUnmapResources(1, &cudaBuffer, 0);

		return nullptr;
	}

}