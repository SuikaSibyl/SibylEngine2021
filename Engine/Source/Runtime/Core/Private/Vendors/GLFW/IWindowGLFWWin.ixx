module;
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <GLFW/glfw3.h>
export module Core.Window.GLFW.Win;
import Core.Window;
import Core.Event;
import Core.Input;
import Core.Window.GLFW;

namespace SIByL
{
	inline namespace Core
	{
		export class IWindowGLFWWin :public IWindowGLFW
		{
		public:
			IWindowGLFWWin(uint32_t const& width, uint32_t const& height, std::string_view name)
				:IWindowGLFW(width, height, name) {}

			virtual auto openFile(const char* filter) noexcept -> std::string override;
			virtual auto saveFile(const char* filter) noexcept -> std::string override;
		};
	}
}
