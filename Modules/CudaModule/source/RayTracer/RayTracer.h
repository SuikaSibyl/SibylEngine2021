#pragma once

namespace SIByL
{
	class CudaSurface;
	class RayTracer
	{
	public:
		static void RenderPtrCudaSurface(CudaSurface* pCudaSurface, float deltaTime);

	};
}