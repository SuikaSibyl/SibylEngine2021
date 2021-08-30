#pragma once

#include "glad/glad.h"
#include "GLFW/glfw3.h"

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
		~OpenGLContext();

	public:
		virtual void Init() override;
		virtual void OnWindowResize(uint32_t width, uint32_t height) override;

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