module;
#include <cstdint>;
#include <functional>
#include <string_view>;
#include <GLFW/glfw3.h>;
export module Core.Window.GLFW;

import Core.Window;
import Core.Input;

namespace SIByL
{
	inline namespace Core
	{
		export class IWindowGLFW :public IWindow
		{
		public:
			IWindowGLFW(uint32_t const& width, uint32_t const& height, std::string_view name);
			~IWindowGLFW();

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
			GLFWwindow* glfw_window;

			struct WindowData
			{
				uint32_t width;
				uint32_t height;
				IWindow* window;
				EventCallbackFn event_callback;
			};

			WindowData data;
			bool is_VSync;
			IInput* input;
			std::string_view name;
		};
	}
}