#include "SIByLpch.h"
#include "OpenGLContext.h"

#include "glad/glad.h"
#include "GLFW/glfw3.h"

namespace SIByL
{
	OpenGLContext::OpenGLContext(GLFWwindow* windowHandle)
		:m_WindowHandle(windowHandle)
	{
		SIByL_CORE_ASSERT(windowHandle, "WindowHandle is NULL!");
	}

	void OpenGLContext::Init()
	{
		// Init GLFW Context
		glfwMakeContextCurrent(m_WindowHandle);

		// Init GLAD
		// OpenGL environment is setted up here
		int status = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
		SIByL_CORE_ASSERT(status, "Failed to initialized Glad!");
	}

	void OpenGLContext::SwipBuffers()
	{
		glfwSwapBuffers(m_WindowHandle);
	}

}