#include "SIByLpch.h"
#include "SRenderContext.h"

#include "Sibyl/Core/Application.h"

namespace SIByL
{
	namespace SRenderPipeline
	{
		unsigned int SRenderContext::ScreenWidth, SRenderContext::ScreenHeight;
		Ref<Scene> SRenderContext::ActiveScene;
		Ref<SPipeline> SRenderContext::ActiveRP;
		Ref<Camera> SRenderContext::ActiveCamera;

		float SRenderContext::GetDelta() {
			return Application::Get().GetFrameTimer()->DeltaTime();
		}

	}
}