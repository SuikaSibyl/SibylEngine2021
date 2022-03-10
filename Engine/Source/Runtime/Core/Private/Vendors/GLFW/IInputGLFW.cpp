module;
#include <GLFW/glfw3.h>
module Core.Input.GLFW;

namespace SIByL::Core
{
	IInputGLFW::IInputGLFW(IWindow* attached_window)
		:attached_window(attached_window)
	{

	}

	auto IInputGLFW::isKeyPressed(int keycode) noexcept -> bool
	{
		auto window = static_cast<GLFWwindow*>(attached_window->getNativeWindow());
		auto state = glfwGetKey(window, keycode);
		return state == GLFW_PRESS || state == GLFW_REPEAT;
	}

	auto IInputGLFW::isMouseButtonPressed(int button) noexcept -> bool
	{
		auto window = static_cast<GLFWwindow*>(attached_window->getNativeWindow());
		auto state = glfwGetMouseButton(window, button);
		return state == GLFW_PRESS;
	}

	auto IInputGLFW::getMousePosition(int button) noexcept -> std::pair<float, float>
	{
		auto window = static_cast<GLFWwindow*>(attached_window->getNativeWindow());
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		return { (float)xpos, (float)ypos };
	}

	auto IInputGLFW::getMouseX() noexcept -> float
	{
		auto window = static_cast<GLFWwindow*>(attached_window->getNativeWindow());
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		return (float)xpos;
	}

	auto IInputGLFW::getMouseY() noexcept -> float
	{
		auto window = static_cast<GLFWwindow*>(attached_window->getNativeWindow());
		double xpos, ypos;
		glfwGetCursorPos(window, &xpos, &ypos);

		return (float)ypos;
	}
}