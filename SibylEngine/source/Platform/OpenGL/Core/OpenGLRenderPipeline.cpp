#include "SIByLpch.h"
#include "OpenGLRenderPipeline.h"

#include "Platform/OpenGL/OpenGLContext.h"
#include "Sibyl/Core/Application.h"

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
		SwapChain* swapChain = OpenGLContext::Get()->GetSwapChain();
		swapChain->SetRenderTarget();
		Application::Get().OnDraw();
		OpenGLContext::Get()->SwipBuffers();
	}
}