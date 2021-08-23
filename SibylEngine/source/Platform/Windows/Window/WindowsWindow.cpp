#include "SIByLpch.h"
#include "WindowsWindow.h"

#include "Sibyl/Events/ApplicationEvent.h"
#include "Sibyl/Events/MouseEvent.h"
#include "Sibyl/Events/KeyEvent.h"

#include "Platform/DirectX12/Renderer/DX12RenderPipeline.h"

namespace SIByL
{
	WindowsWindow* WindowsWindow::Main;

	WindowsWindow::WindowsWindow(const WindowProps& props)
	{
		if (Main == nullptr) Main = this;
		Init(props);
	}

	WindowsWindow::~WindowsWindow()
	{
		Shutdown();
	}

	LRESULT CALLBACK
		MainWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
	{
		// Forward hwnd on because we can get messages (e.g., WM_CREATE)
		// before CreateWindow returns, and thus before mhMainWnd is valid.
		return WindowsWindow::Main->MsgProc(hwnd, msg, wParam, lParam);
	}
	
	LRESULT CALLBACK WindowsWindow::MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
	{
		if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wParam, lParam))
			return true;

		switch (msg)
		{
		// WM_ACTIVATE is sent when the window is activated or deactivated.  
		// We pause the game when the window is deactivated and unpause it 
		// when it becomes active. 
		case WM_ACTIVATE:
			return 0;
		// WM_SIZE is sent when the user resizes the window.
		case WM_SIZE:
		{
			m_Data.Width = LOWORD(lParam);
			m_Data.Height = HIWORD(lParam);
			WindowResizeEvent event(m_Data.Width, m_Data.Height);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		// WM_EXITSIZEMOVE is sent when the user grabs the resize bars.
		case WM_ENTERSIZEMOVE:
			return 0;
		// WM_EXITSIZEMOVE is sent when the user releases the resize bars.
		// Here we reset everything based on the new window dimensions.
		case WM_EXITSIZEMOVE:
			return 0;
		// WM_DESTROY is sent when the window is being destroyed.
		case WM_DESTROY:
		{
			PostQuitMessage(0);
			WindowCloseEvent event;
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_CHAR:
		{
			KeyTypedEvent event((unsigned int)wParam);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		// The WM_MENUCHAR message is sent when a menu is active and the user presses 
		// a key that does not correspond to any mnemonic or accelerator key. 
		case WM_MENUCHAR:
			return MAKELRESULT(0, MNC_CLOSE);
		// Catch this message so to prevent the window from becoming too small.
		case WM_GETMINMAXINFO:
			return 0;
		case WM_LBUTTONDOWN:
		{
			MouseButtonPressedEvent event(0);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_MBUTTONDOWN:
		{
			MouseButtonPressedEvent event(2);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_RBUTTONDOWN:
		{
			MouseButtonPressedEvent event(1);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_LBUTTONUP:
		{
			MouseButtonReleasedEvent event(0);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_MBUTTONUP:
		{
			MouseButtonReleasedEvent event(2);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_RBUTTONUP:
		{
			MouseButtonReleasedEvent event(1);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_MOUSEMOVE:
		{
			MouseMovedEvent event((float)GET_X_LPARAM(lParam), (float)GET_Y_LPARAM(lParam));
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_MOUSEWHEEL:
		{
			MouseScrolledEvent event(0, (float)GET_WHEEL_DELTA_WPARAM(wParam) / 120);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_KEYDOWN:
		{
			int previousDown = ((lParam >> 30) & 1) == 1;
			KeyPressedEvent event((int)wParam, previousDown);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}
		case WM_KEYUP:
			KeyReleasedEvent event((int)wParam);
			if (m_Data.EventCallback == nullptr) return 0;
			m_Data.EventCallback(event);
			return 0;
		}

		return DefWindowProc(hwnd, msg, wParam, lParam);
	}

	void WindowsWindow::Init(const WindowProps& props)
	{
		m_Data.Title = props.Title;
		m_Data.Width = props.Width;
		m_Data.Height = props.Height;

		SIByL_CORE_INFO("Creating windows {0} ({1}, {2})", props.Title, props.Width, props.Height);

		#if defined(DEBUG) | defined(_DEBUG)
				_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
		#endif

		WNDCLASS wc;
		wc.style = CS_HREDRAW | CS_VREDRAW;
		wc.lpfnWndProc = MainWndProc;
		wc.cbClsExtra = 0;
		wc.cbWndExtra = 0;
		wc.hInstance = mhAppInst;
		wc.hIcon = LoadIcon(0, IDI_APPLICATION);
		wc.hCursor = LoadCursor(0, IDC_ARROW);
		wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
		wc.lpszMenuName = 0;
		wc.lpszClassName = L"MainWnd";

		if (!RegisterClass(&wc))
		{
			SIByL_CORE_ERROR("Windows Window RegisterClass Failed.");
			return;
		}

		int syswidth = GetSystemMetrics(SM_CXSCREEN);
		int sysheight = GetSystemMetrics(SM_CYSCREEN);

		// Compute window rectangle dimensions based on requested client area dimensions.
		RECT R = { 0, 0, (LONG)props.Width, (LONG)props.Height };
		AdjustWindowRect(&R, WS_OVERLAPPEDWINDOW, false);
		int width = R.right - R.left;
		int height = R.bottom - R.top;
		SIByL_CORE_INFO("WindowsWindow Init finished");

		std::wstring_convert<std::codecvt<wchar_t, char, mbstate_t>> converter(new std::codecvt<wchar_t, char, mbstate_t>("CHS"));
		std::wstring mMainWndCaption = converter.from_bytes(props.Title);
		mhMainWnd = CreateWindow(L"MainWnd", mMainWndCaption.c_str(),
			WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, width, height, 0, 0, mhAppInst, 0);
		if (!mhMainWnd)
		{
			SIByL_CORE_ERROR("Windows Window CreateWindow Failed.");
			return;
		}

		// Init DX12 Environment
		m_DX12Env = std::make_unique<DX12Context>();
		m_DX12Env->Init();
		m_GraphicContext = m_DX12Env.get();

		ShowWindow(mhMainWnd, SW_SHOW);

		UpdateWindow(mhMainWnd);

		SetVSync(true);
	}

	void WindowsWindow::Shutdown()
	{
		DestroyWindow(mhMainWnd);
	}

	void* WindowsWindow::GetNativeWindow() const
	{
		return (void*)mhMainWnd;
	}

	void WindowsWindow::OnUpdate()
	{
		MSG msg = { 0 };

		// If there are Window messages then process them.
		if (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		// Otherwise, do animation/game stuff.
		DX12RenderPipeline::DrawFrame();
	}

	void WindowsWindow::SetVSync(bool enabled)
	{

		m_Data.VSync = enabled;
	}

	bool WindowsWindow::IsVSync() const
	{
		return m_Data.VSync;
	}
}