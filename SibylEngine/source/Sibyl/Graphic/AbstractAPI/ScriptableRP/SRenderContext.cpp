#include "SIByLpch.h"
#include "SRenderContext.h"

namespace SIByL
{
	namespace SRenderPipeline
	{
		unsigned int SRenderContext::ScreenWidth, SRenderContext::ScreenHeight;
		Ref<Scene> SRenderContext::ActiveScene;
		Ref<SPipeline> SRenderContext::ActiveRP;
		Ref<Camera> SRenderContext::ActiveCamera;
	}
}