module;
#include <vector>
export module Core.Input;

namespace SIByL
{
	inline namespace Core
	{
		export class IInput
		{
		public:
			virtual auto isKeyPressed(int keycode) noexcept -> bool = 0;
			virtual auto isMouseButtonPressed(int button) noexcept -> bool = 0;
			virtual auto getMousePosition(int button) noexcept -> std::pair<float, float> = 0;
			virtual auto getMouseX() noexcept -> float = 0;
			virtual auto getMouseY() noexcept -> float = 0;
		};
	}
}