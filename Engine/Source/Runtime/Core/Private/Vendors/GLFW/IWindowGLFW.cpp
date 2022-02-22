module;
#include <cstdint>
#include <functional>
#include <string_view>
#include <GLFW/glfw3.h>
module Core.Window.GLFW;

import Core.Log;
import Core.Assert;
import Core.Event;
import Core.Input.GLFW;

namespace SIByL::Core
{
	static bool g_glfw_initialzied = false;
	static unsigned int g_glfw_count = 0;

	void GLFWErrorCallback(int error, const char* description)
	{
		SE_CORE_ERROR("GLFW Error ({0}): {1}", error, description);
	}

	IWindowGLFW::IWindowGLFW(uint32_t const& width, uint32_t const& height, std::string_view name)
		:data{ width, height, (IWindow*)this }, name{ name }
	{
		if (!g_glfw_initialzied)
		{
			int success = glfwInit();
			SE_CORE_ASSERT(success, "Could not initialzie GLWF!");
			glfwSetErrorCallback(GLFWErrorCallback);
			g_glfw_initialzied = true;
		}
		g_glfw_count++;
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		
		glfw_window = glfwCreateWindow(width, height, name.data(), nullptr, nullptr);
		glfwSetWindowUserPointer(glfw_window, &data);
		input = new IInputGLFW(this);

		//setVSync(isVSync);

		// Set GLFW Callbacks
		glfwSetWindowSizeCallback(glfw_window, [](GLFWwindow* window, int width, int height)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				WindowResizeEvent event(width, height, (void*)data.window);
				data.width = width;
				data.height = height;
				data.event_callback(event);
			});

		glfwSetWindowCloseCallback(glfw_window, [](GLFWwindow* window)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				WindowCloseEvent event((void*)data.window);
				data.event_callback(event);
			});

		glfwSetKeyCallback(glfw_window, [](GLFWwindow* window, int key, int scancode, int action, int mods)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				switch (action)
				{
				case GLFW_PRESS:
				{
					KeyPressedEvent event(key, 0);
					data.event_callback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					KeyReleasedEvent event(key);
					data.event_callback(event);
					break;
				}
				case GLFW_REPEAT:
				{
					KeyPressedEvent event(key, 1);
					data.event_callback(event);
					break;
				}
				default:
					break;
				}
			});

		glfwSetCharCallback(glfw_window, [](GLFWwindow* window, unsigned int keycode)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				KeyTypedEvent event(keycode);
				data.event_callback(event);
			});

		glfwSetMouseButtonCallback(glfw_window, [](GLFWwindow* window, int button, int action, int mods)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				switch (action)
				{
				case GLFW_PRESS:
				{
					MouseButtonPressedEvent event(button);
					data.event_callback(event);
					break;
				}
				case GLFW_RELEASE:
				{
					MouseButtonReleasedEvent event(button);
					data.event_callback(event);
					break;
				}
				default:
					break;
				}
			});

		glfwSetScrollCallback(glfw_window, [](GLFWwindow* window, double xOffset, double yOffset)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				MouseScrolledEvent event((float)xOffset, (float)yOffset);
				data.event_callback(event);
			});

		glfwSetCursorPosCallback(glfw_window, [](GLFWwindow* window, double xPos, double yPos)
			{
				WindowData& data = *(WindowData*)glfwGetWindowUserPointer(window);
				MouseMovedEvent event((float)xPos, (float)yPos);
				data.event_callback(event);
			});
	}

	IWindowGLFW::~IWindowGLFW()
	{
		delete input;
		glfwDestroyWindow(glfw_window);
		g_glfw_count--;

		if (g_glfw_count <= 0)
		{
			glfwTerminate();
			g_glfw_initialzied = false;
		}
	}

	auto IWindowGLFW::onUpdate() noexcept -> void
	{
		glfwPollEvents();
	}

	auto IWindowGLFW::getWidth() const noexcept -> unsigned int
	{
		return data.width;
	}

	auto IWindowGLFW::getHeight() const noexcept -> unsigned int
	{
		return data.height;
	}

	auto IWindowGLFW::getHighDPI() const noexcept -> float
	{
		float x, y;
		glfwGetWindowContentScale(glfw_window, &x, &y);
		return x;
	}

	auto IWindowGLFW::getNativeWindow() const noexcept -> void* 
	{
		return (void*)glfw_window;
	}

	auto IWindowGLFW::setVSync(bool enabled) noexcept -> void
	{
		if (enabled)
			glfwSwapInterval(1);
		else
			glfwSwapInterval(0);

		is_VSync = enabled;
	}

	auto IWindowGLFW::isVSync() const noexcept -> bool
	{
		return is_VSync;
	}

	auto IWindowGLFW::setEventCallback(const EventCallbackFn& callback) noexcept -> void
	{
		data.event_callback = callback;
	}

	auto IWindowGLFW::getInput() const noexcept -> IInput*
	{
		return input;
	}
	
	auto IWindowGLFW::waitUntilNotMinimized(unsigned int& width, unsigned int height) const noexcept -> void
	{
		int _width = (int)width;
		int _height = (int)height;
		while (_width == 0 || _height == 0) {
			glfwGetFramebufferSize(glfw_window, &_width, &_height);
			glfwWaitEvents();
		}
		width = static_cast<uint32_t>(_width);
		height = static_cast<uint32_t>(_height);
	}

	auto IWindowGLFW::getFramebufferSize(uint32_t& width, uint32_t& height) const noexcept -> void
	{
		int _width, _height;
		glfwGetFramebufferSize(glfw_window, &_width, &_height);
		width = static_cast<uint32_t>(_width);
		height = static_cast<uint32_t>(_height);
	}

}