#include "CudaModulePCH.h"
#include "RayTracerInterface.h"

#include "GraphicInterop/TextureInterface.h"
#include "GraphicInterop/CudaSurface.h"
#include "RayTracer/RayTracer.h"


namespace SIByL
{
	void CUDARayTracerInterface::RenderPtrCudaSurface(PtrCudaSurface* surface, float deltaTime)
	{
		surface->StartOpenGLMapping();
		RayTracer::RenderPtrCudaSurface(surface->pCudaSurface, deltaTime);
		surface->EndOpenGLMapping();
	}
}