#pragma once

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/GraphicContext.h"
#include "Platform/OpenGL/AbstractAPI/Top/OpenGLRenderPipeline.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLSwapChain.h"

struct GLFWwindow;

namespace SIByL
{
	class OpenGLContext :public GraphicContext
	{
	public:
		OpenGLContext(GLFWwindow* windowHandle);
		~OpenGLContext();

		virtual void StartCommandList() override;
		virtual void EndCommandList() override;

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