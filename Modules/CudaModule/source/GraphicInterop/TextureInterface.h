#pragma once

namespace SIByL
{
	class CudaTexture;
	class PtrCudaTexture
	{
	public:
		PtrCudaTexture();
		CudaTexture* pCudaTexture;
		void SetFromOpenGLTexture(uint32_t id, uint32_t width, uint32_t height);
	};

	class CudaSurface;
	class PtrCudaSurface
	{
	public:
		PtrCudaSurface();
		CudaSurface* pCudaSurface;
		void RegisterByOpenGLTexture(uint32_t id, uint32_t width, uint32_t height);
		void StartOpenGLMapping();
		void EndOpenGLMapping();
	};
}