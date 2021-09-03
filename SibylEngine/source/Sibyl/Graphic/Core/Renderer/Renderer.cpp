#include "SIByLpch.h"
#include "Renderer.h"

namespace SIByL
{
	RasterRenderer Renderer::s_Raster = RasterRenderer::OpenGL;
	RayTracerRenderer Renderer::s_RayTracer = RayTracerRenderer::Cuda;

}