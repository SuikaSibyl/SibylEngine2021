export module Core.Input.GLFW;

import Core.Window;
import Core.Input;

namespace SIByL
{
	inline namespace Core
	{
		export class IInputGLFW : public IInput
		{
		public:
			IInputGLFW(IWindow* attached_window);

			virtual auto isKeyPressed(CodeEnum const& keycode) noexcept -> bool override;
			virtual auto isMouseButtonPressed(CodeEnum const& button) noexcept -> bool override;
			virtual auto getMousePosition(int button) noexcept -> std::pair<float, float> override;
			virtual auto getMouseX() noexcept -> float override;
			virtual auto getMouseY() noexcept -> float override;
			virtual auto getMouseScrollX() noexcept -> float override { return scrollX; }
			virtual auto getMouseScrollY() noexcept -> float override { return scrollY; }

			virtual auto disableCursor() noexcept -> void override;
			virtual auto enableCursor() noexcept -> void override;

			float scrollX = 0;
			float scrollY = 0;

		private:
			IWindow* attached_window;
		};

	}
}