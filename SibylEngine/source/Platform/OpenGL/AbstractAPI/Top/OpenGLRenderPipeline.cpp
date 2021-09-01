#include "SIByLpch.h"
#include "OpenGLRenderPipeline.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"
#include "Sibyl/Core/Application.h"
#include "Sibyl/ImGui/ImGuiLayer.h"

namespace SIByL
{
	OpenGLRenderPipeline* OpenGLRenderPipeline::Main;

	OpenGLRenderPipeline::OpenGLRenderPipeline()
	{
		SIByL_CORE_ASSERT(!Main, "OGL Render Pipeline Already Exists!");
		Main = this;
	}

	void OpenGLRenderPipeline::DrawFrameImpl()
	{
		PROFILE_SCOPE_FUNCTION();

		Ref<SwapChain> swapChain = OpenGLContext::Get()->GetSwapChain();

		// Bind Swap Chain as Render Target
		// -------------------------------------
		swapChain->SetRenderTarget();

		// Drawcalls
		Application::Get().OnDraw();

		swapChain->PreparePresent();
		
		ImGuiLayer::OnDrawAdditionalWindows();

		swapChain->Present();
	}
}