#pragma once

#include "CudaModulePCH.h"

namespace SIByL
{
	class CudaSurface
	{
	public:
		CudaSurface() = default;
		CudaSurface(const std::string& path);
		CudaSurface(size_t width, size_t height);
		cudaSurfaceObject_t GetSurfaceObject();

		void RegisterByOpenGLTexture(uint32_t id, uint32_t width, uint32_t height);
		void StartOpenGLMapping();
		void EndOpenGLMapping();

		uint32_t GetWidth() const { return Width; }
		uint32_t GetHeight() const { return Height; }

	private:
		uint32_t Width, Height;
		cudaArray_t cuArray;
		cudaGraphicsResource_t cudaBuffer;
		cudaSurfaceObject_t mSurfaceObject = 0;
	};
}