module;
#include <cstdint>;
#include <functional>
#include <string_view>;
#include <Windows.h>
export module Core.Window.WIN;

import Core.Window;
import Core.Input;

namespace SIByL
{
	inline namespace Core
	{
		export class IWindowWIN :public IWindow
		{
		public:
			IWindowWIN(uint32_t const& width, uint32_t const& height, std::string_view name);
			~IWindowWIN();

			virtual auto onUpdate() noexcept -> void override;
			virtual auto getWidth() const noexcept -> unsigned int override;
			virtual auto getHeight() const noexcept -> unsigned int override;
			virtual auto getHighDPI() const noexcept -> float override;
			virtual auto getNativeWindow() const noexcept -> void* override;
			virtual auto setVSync(bool enabled) noexcept -> void override;
			virtual auto isVSync() const noexcept -> bool override;
			virtual auto setEventCallback(const EventCallbackFn& callback) noexcept -> void override;
			virtual auto getInput() const noexcept -> IInput* override;

		private:
			struct WindowData
			{
				uint32_t width;
				uint32_t height;
				EventCallbackFn event_callback;
			};

			bool is_VSync;
			IInput* input;
			std::string_view name;

			WindowData	data;
			HINSTANCE	mhAppInst = nullptr; // application instance handle
			HWND		mhMainWnd = nullptr; // main window handle

			auto getHWND() -> HWND* { return &mhMainWnd; }
			LRESULT CALLBACK MsgProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);

		};
	}
}