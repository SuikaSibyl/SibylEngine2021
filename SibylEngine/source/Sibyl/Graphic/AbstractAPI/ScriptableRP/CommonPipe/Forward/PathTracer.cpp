#include "SIByLpch.h"
#include "PathTracer.h"

#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"
#include "Sibyl/ECS/Scene/Scene.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/FrameConstantsManager.h"


#include "CudaModule/source/RayTracer/RayTracerInterface.h"

namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipePathTracer::Build()
		{
			FrameBufferDesc desc;
			desc.Width = 1280;
			desc.Height = 720;
			desc.Formats = {
				FrameBufferTextureFormat::RGB8,		// Color
			 };
			// Frame Buffer 0: Main Render Buffer
			mFrameBuffer = FrameBuffer::Create(desc, "PathTracer");
			mFrameBuffer->SetClearColor({ 0.1, 0.1, 0.1, 0.05 });
			mFrameBuffer->GetRenderTarget(0)->InvalidCudaSurface();
		}

		void SRPPipePathTracer::Attach()
		{

		}

		void SRPPipePathTracer::Draw()
		{
#ifdef SIBYL_PLATFORM_CUDA
			CUDARayTracerInterface::RenderPtrCudaSurface(mFrameBuffer->GetRenderTarget(0)->GetCudaSurface(), SRenderContext::GetDelta());
#endif // SIBYL_PLATFORM_CUDA
		}

		void SRPPipePathTracer::DrawImGui()
		{

		}

		RenderTarget* SRPPipePathTracer::GetRenderTarget(const std::string& name)
		{
			if (name == "Color")
			{
				return mFrameBuffer->GetRenderTarget(0);
			}

			SIByL_CORE_ERROR("Path Tracer Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipePathTracer::SetInput(const std::string& name, RenderTarget* target)
		{

		}
	}
}