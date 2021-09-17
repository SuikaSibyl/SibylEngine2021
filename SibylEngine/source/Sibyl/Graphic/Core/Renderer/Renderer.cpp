#include "SIByLpch.h"
#include "Renderer.h"

#include "Platform/Cuda/Common/SCudaContext.h"

namespace SIByL
{
	RasterRenderer Renderer::s_Raster = RasterRenderer::OpenGL;
	RayTracerRenderer Renderer::s_RayTracer = RayTracerRenderer::Cuda;

	void Renderer::SetRayTracer(RayTracerRenderer rayTracer)
	{
		s_RayTracer = rayTracer; 
		if (rayTracer == RayTracerRenderer::Cuda)
		{
			SCudaContext::Init();
		}
	}
}