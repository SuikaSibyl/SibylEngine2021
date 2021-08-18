#pragma once

#include "Sibyl/Renderer/GraphicContext.h"
#include "Platform/OpenGL/Core/OpenGLRenderPipeline.h"
#include "Platform/OpenGL/Renderer/OpenGLSwapChain.h"

struct GLFWwindow;

namespace SIByL
{
	class OpenGLContext :public GraphicContext
	{
	public:
		OpenGLContext(GLFWwindow* windowHandle);

		virtual void Init() override;
		virtual void SwipBuffers() override;
		static inline OpenGLContext* Get() { return Main; }

		SwapChain* GetSwapChain() { return m_SwapChain.get(); }
		
	private:
		GLFWwindow* m_WindowHandle;
		std::unique_ptr<OpenGLRenderPipeline> m_RenderPipeline;
		std::unique_ptr<OpenGLSwapChain> m_SwapChain;
		static OpenGLContext* Main;
	};
}