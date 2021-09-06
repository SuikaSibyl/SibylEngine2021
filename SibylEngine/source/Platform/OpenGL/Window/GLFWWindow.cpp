#include "SIByLpch.h"
#include <glad/glad.h> 
#include "GLFWWindow.h"

#include "Sibyl/Core/Events/ApplicationEvent.h"
#include "Sibyl/Core/Events/MouseEvent.h"
#include "Sibyl/Core/Events/KeyEvent.h"
#include "Sibyl/Core/Input.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"

namespace SIByL
{
	static bool s_GLFWInitialzied = false;
	GLFWWindow* GLFWWindow::Main;

	static void GLFWErrorCallback(int error, const char* description)
	{
		SIByL_CORE_ERROR("GLFW Error ({0}): {1}", error, description);

	}

	GLFWWindow::GLFWWindow(const WindowProps& props)
	{
		PROFILE_SCOPE_FUNCTION();

		SIByL_CORE_ASSERT(!Main, "GLFW Window Already Exists!");
		Main = this;
		Init(props);
	}

	GLFWWindow::~GLFWWindow()
	{
		Shutdown();
	}

	void GLFWWindow::Init(const WindowProps& props)
	{
		m_Data.Title = props.Title;
		m_Data.Width = props.Width;
		m_Data.Height = props.Height;

		SIByL_CORE_INFO("Creating windows {0} ({1}, {2})", props.Title, props.Width, props.Height);

		if (!s_GLFWInitialzied)
		{
			int success = glfwInit();
			SIByL_CORE_ASSERT(success, "Could not initialzie GLWF!");
			glfwSetErrorCallback(GLFWErrorCallback);
			s_GLFWInitialzied = true;
		}

		m_Window = glfwCreateWindow((int)props.Width, (int)props.Height, m_Data.Title.c_str(), nullptr, nullptr);
		
		m_OpenGLContext = std::make_unique<OpenGLContext>(m_Window);
		m_OpenGLContext->Init();
		m_GraphicContext = m_OpenGLContext.get();

		glfwSetWindowUserPointer(m_Window, &m_Data);
		SetVSync(false);

		// Set GLFW Callbacks
		glfwSetWindowSizeCallback(m_Window, [](GLFWwindow* window, int width, int height)
			{
				WindowData& data = *(WindowData*) glfwGetWindowUserPointer(window);
				WindowResizeEvent event(width, height);
				data.Width = width;
				data.Height = height;
				data.EventCallback(event);
			});

		glfwSetWindowCloseCallback(m_Window, [](GLFWwindow* window)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				WindowCloseEvent event;
				data.EventCallback(event);
			});

		glfwSetKeyCallback(m_Window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				switch (action)
				{
				case GLFW_PRESS:
				{
					KeyPressedEvent event(key, 0);
					data.EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					KeyReleasedEvent event(key);
					data.EventCallback(event);
					break;
				}
				case GLFW_REPEAT:
				{
					KeyPressedEvent event(key, 1);
					data.EventCallback(event);
					break;
				}
				default:
					break;
				}
			});

		glfwSetCharCallback(m_Window, [](GLFWwindow* window, unsigned int keycode)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				KeyTypedEvent event(keycode);
				data.EventCallback(event);
			});

		glfwSetMouseButtonCallback(m_Window, [](GLFWwindow* window, int button, int action, int mods)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				switch (action)
				{
				case GLFW_PRESS:
				{
					MouseButtonPressedEvent event(button);
					data.EventCallback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					MouseButtonReleasedEvent event(button);
					data.EventCallback(event);
					break;
				}
				default:
					break;
				}
			});

		glfwSetScrollCallback(m_Window, [](GLFWwindow* window, double xOffset, double yOffset)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				MouseScrolledEvent event((float)xOffset, (float)yOffset);
				data.EventCallback(event);
			});

		glfwSetCursorPosCallback(m_Window, [](GLFWwindow* window, double xPos, double yPos)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				MouseMovedEvent event((float)xPos, (float)yPos);
				data.EventCallback(event);
			});
	}

	void GLFWWindow::Shutdown()
	{
		PROFILE_SCOPE_FUNCTION();

		glfwDestroyWindow(m_Window);
	}

	void GLFWWindow::OnUpdate()
	{
		PROFILE_SCOPE_FUNCTION();

		glfwPollEvents();
		OpenGLRenderPipeline::DrawFrame();
	}

	void GLFWWindow::SetVSync(bool enabled)
	{
		PROFILE_SCOPE_FUNCTION();

		if (enabled)
			glfwSwapInterval(1);
		else
			glfwSwapInterval(0);

		m_Data.VSync = enabled;
	}

	bool GLFWWindow::IsVSync() const
	{
		return m_Data.VSync;
	}

	void* GLFWWindow::GetNativeWindow() const
	{
		return (void*)m_Window;
	}

	float GLFWWindow::GetHighDPI()
	{
		float x, y;
		glfwGetWindowContentScale(m_Window, &x, &y);
		return x;
	}
}