#include "SIByLpch.h"
#include "WindowsInput.h"

#include "Sibyl/Core/Application.h"

namespace SIByL
{
	bool WindowsInput::IsKeyPressedImpl(int keycode)
	{
		//auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		//auto state = glfwGetKey(window, keycode);
		//return state == GLFW_PRESS || state == GLFW_REPEAT;

		return false;
	}
	bool WindowsInput::IsMouseButtonPressedImpl(int button)
	{
		//auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		//auto state = glfwGetMouseButton(window, button);
		//return state == GLFW_PRESS;

		return false;
	}

	std::pair<float, float> WindowsInput::GetMousePositionImpl()
	{
		//auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		//double xpos, ypos;
		//glfwGetCursorPos(window, &xpos, &ypos);

		return { 0.0f, 0.0f };
	}

	float WindowsInput::GetMouseXImpl()
	{
		//auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		//double xpos, ypos;
		//glfwGetCursorPos(window, &xpos, &ypos);

		return 0.0f;
	}

	float WindowsInput::GetMouseYImpl()
	{
		//auto window = static_cast<GLFWwindow*>(Application::Get().GetWindow().GetNativeWindow());
		//double xpos, ypos;
		//glfwGetCursorPos(window, &xpos, &ypos);

		return 0.0f;
	}
}