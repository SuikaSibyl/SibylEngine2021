#pragma once

#include "CudaModulePCH.h"

namespace SIByL
{
	class CudaTexture
	{
	public:
		CudaTexture() = default;
		CudaTexture(const std::string& path);
		cudaTextureObject_t GetTextureObject();
		
		CudaTexture* CreateFromOpenGLTexture(uint32_t id, uint32_t width, uint32_t height);


	private:
		cudaTextureObject_t mTextureObject = 0;
	};
}