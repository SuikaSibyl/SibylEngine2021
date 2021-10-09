#include "CudaModulePCH.h"
#include "CudaTexture.h"
#include "CudaSurface.h"
#include "TextureInterface.h"

namespace SIByL
{
	//////////////////////////////////////////////////////
	///					CUDA Texture				   ///
	//////////////////////////////////////////////////////

	PtrCudaTexture::PtrCudaTexture()
	{
		pCudaTexture = new CudaTexture();
	}

	void PtrCudaTexture::SetFromOpenGLTexture(uint32_t id, uint32_t width, uint32_t height)
	{
		pCudaTexture->CreateFromOpenGLTexture(id, width, height);
	}

	//////////////////////////////////////////////////////
	///					CUDA Surface				   ///
	//////////////////////////////////////////////////////

	PtrCudaSurface::PtrCudaSurface()
	{
		pCudaSurface = new CudaSurface();
	}

	void PtrCudaSurface::RegisterByOpenGLTexture(uint32_t id, uint32_t width, uint32_t height)
	{
		pCudaSurface->RegisterByOpenGLTexture(id, width, height);
	}

	void PtrCudaSurface::StartOpenGLMapping()
	{
		pCudaSurface->StartOpenGLMapping();
	}

	void PtrCudaSurface::EndOpenGLMapping()
	{
		pCudaSurface->EndOpenGLMapping();
	}

}