#include "SIByLpch.h"
#include "OpenGLContext.h"

#include "glad/glad.h"
#include "GLFW/glfw3.h"

#include "Sibyl/Core/Application.h"
#include "Platform/OpenGL/AbstractAPI/Bottom/OpenGLSynchronizer.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLCommandList.h"

namespace SIByL
{
	OpenGLContext* OpenGLContext::Main;

	OpenGLContext::OpenGLContext(GLFWwindow* windowHandle)
		:m_WindowHandle(windowHandle)
	{
		PROFILE_SCOPE_FUNCTION();

		SIByL_CORE_ASSERT(windowHandle, "WindowHandle is NULL!");
		SIByL_CORE_ASSERT(!Main, "OpenGL Context Already Exists!");
		Main = this;

		// Create Command List
		m_CommandList = std::make_shared<OpenGLCommandList>();
		m_Synchronizer.reset(new OpenGLSynchronizer());
	}

	OpenGLContext::~OpenGLContext()
	{
		PROFILE_SCOPE_FUNCTION();

		Application::Get().OnResourceDestroy();
	}

	void OpenGLContext::Init()
	{
		// Init GLFW Context
		glfwMakeContextCurrent(m_WindowHandle);

		// Init GLAD
		// OpenGL environment is setted up here
		int status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
		SIByL_CORE_ASSERT(status, "Failed to initialized Glad!");

		m_RenderPipeline = std::make_unique<OpenGLRenderPipeline>();
		m_SwapChain = std::make_unique<OpenGLSwapChain>();
	}

	void OpenGLContext::OnWindowResize(uint32_t width, uint32_t height)
	{
		PROFILE_SCOPE_FUNCTION();

		m_SwapChain->Reisze(width, height);
	}
}