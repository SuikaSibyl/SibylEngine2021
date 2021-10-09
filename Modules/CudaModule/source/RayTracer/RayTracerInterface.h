#pragma once

namespace SIByL
{
	class PtrCudaTexture;
	class PtrCudaSurface;
	class CUDARayTracerInterface
	{
	public:
		static void RenderPtrCudaSurface(PtrCudaSurface* texture, float deltaTime);

	private:

	};
}