#pragma once
#include "Sibyl/Core/Window.h"
#include "Platform/DirectX12/Common/DX12Context.h"

namespace SIByL
{
	class WindowsWindow :public Window
	{
	public:
		static WindowsWindow* Main;
		WindowsWindow(const WindowProps& props);
		virtual ~WindowsWindow();

		void OnUpdate() override;

		inline unsigned int GetWidth() const override { return m_Data.Width; }
		inline unsigned int GetHeight() const override { return m_Data.Height; }

		// Window attributes
		inline void SetEventCallback(const EventCallbackFn& callback) override
		{
			m_Data.EventCallback = callback;
		}

		void SetVSync(bool enabled) override;
		bool IsVSync() const override;

		virtual void* GetNativeWindow() const override;

	private:
		virtual void Init(const WindowProps& props);
		virtual void Shutdown();

	private:

		struct WindowData
		{
			std::string Title;
			unsigned int Width, Height;
			bool VSync;

			EventCallbackFn EventCallback;
		};

		WindowData m_Data;

	protected:
		HINSTANCE mhAppInst = nullptr; // application instance handle
		HWND      mhMainWnd = nullptr; // main window handle
	public:
		LRESULT CALLBACK MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
		HWND* GetHWND() { return &mhMainWnd; }
	private:
		std::unique_ptr<DX12Context> m_DX12Env;
	};
}