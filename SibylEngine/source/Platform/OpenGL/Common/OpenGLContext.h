#pragma once

#include "Sibyl/Renderer/GraphicContext.h"
#include "Platform/OpenGL/Renderer/OpenGLRenderPipeline.h"
#include "Platform/OpenGL/Renderer/OpenGLSwapChain.h"

struct GLFWwindow;

namespace SIByL
{
	class OpenGLContext :public GraphicContext
	{
	public:
		OpenGLContext(GLFWwindow* windowHandle);

	public:
		virtual void Init() override;
		
	private:
		GLFWwindow* m_WindowHandle;
		std::unique_ptr<OpenGLRenderPipeline> m_RenderPipeline;

		// Get Context
	public:
		static inline OpenGLContext* Get() { return Main; }
	private:
		static OpenGLContext* Main;
	};
}