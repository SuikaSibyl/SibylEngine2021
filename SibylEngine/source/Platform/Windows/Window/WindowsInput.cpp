#include "SIByLpch.h"
#include "WindowsInput.h"

#include "Sibyl/Core/Application.h"
#include "Platform/Windows/Window/WindowsWindow.h"

namespace SIByL
{
	bool WindowsInput::IsKeyPressedImpl(int keycode)
	{
		if (GetAsyncKeyState(keycode) & 0x8000)
		{
			return true;
		}

		return false;
	}
	bool WindowsInput::IsMouseButtonPressedImpl(int button)
	{
		if (GetAsyncKeyState(button) & 0x8000)
		{
			return true;
		}

		return false;
	}

	std::pair<float, float> WindowsInput::GetMousePositionImpl()
	{
		POINT pt;
		BOOL bReturn = GetCursorPos(&pt);
		if (bReturn != 0)
		{
			if (ScreenToClient((HWND)WindowsWindow::Main->GetNativeWindow(), &pt))
			{
				return { pt.x ,pt.y };
			}
		}
		return { 0.0f, 0.0f };
	}

	float WindowsInput::GetMouseXImpl()
	{
		POINT pt;
		BOOL bReturn = GetCursorPos(&pt);
		if (bReturn != 0)
		{
			if (ScreenToClient((HWND)WindowsWindow::Main->GetNativeWindow(), &pt))
			{
				return pt.x;
			}
		}

		return 0.0f;
	}

	float WindowsInput::GetMouseYImpl()
	{
		POINT pt;
		BOOL bReturn = GetCursorPos(&pt);
		if (bReturn != 0)
		{
			if (ScreenToClient((HWND)WindowsWindow::Main->GetNativeWindow(), &pt))
			{
				return pt.y;
			}
		}

		return 0.0f;
	}
}