module;
#include <cstdint>
#include <functional>
#include <string_view>
#include <Windows.h>
module Core.Window.WIN;

import Core.Log;
//import Core.Assert;
//import Core.Event;

namespace SIByL::Core
{
	LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
	{
		// Forward hwnd on because we can get messages (e.g., WM_CREATE)
		// before CreateWindow returns, and thus before hWnd is valid.
		switch (msg)
		{
		case WM_DESTROY:
			PostQuitMessage(0);
			return 0L;
		case WM_LBUTTONDOWN:
			
		default:
			return DefWindowProc(hwnd, msg, wParam, lParam);
		}

	}

	IWindowWIN::IWindowWIN(uint32_t const& width, uint32_t const& height, std::string_view name)
		:data{ width, height }, name{ name }
	{
		//WNDCLASS wc;
		//wc.style = CS_HREDRAW | CS_VREDRAW;
		//wc.lpfnWndProc = WndProc;
		//wc.cbClsExtra = 0;
		//wc.cbWndExtra = 0;
		//wc.hInstance = mhAppInst;
		//wc.hIcon = LoadIcon(0, IDI_APPLICATION);
		//wc.hCursor = LoadCursor(0, IDC_ARROW);
		//wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
		//wc.lpszMenuName = 0;
		//wc.lpszClassName = name;


	}

}