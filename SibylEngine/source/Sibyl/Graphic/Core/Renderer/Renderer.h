#pragma once

namespace SIByL
{
	enum class RasterRenderer
	{
		OpenGL,
		DirectX12,
		CpuSoftware,
		GpuSoftware,
	};

	enum class RayTracerRenderer
	{
		CpuSoftware,
		Cuda,
		DXR,
	};

	class Renderer
	{
	public:
		static inline void SetRaster(RasterRenderer raster) { s_Raster = raster; }
		static inline void SetRayTracer(RayTracerRenderer rayTracer) { s_RayTracer = rayTracer; }

		static inline RasterRenderer GetRaster() { return s_Raster; }
		static inline RayTracerRenderer GetRayTracer() { return s_RayTracer; }

	private:
		static RasterRenderer s_Raster;
		static RayTracerRenderer s_RayTracer;
	};

}