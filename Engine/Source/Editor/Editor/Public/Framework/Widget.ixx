export module Editor.Widget;

namespace SIByL::Editor
{
	export struct Widget
	{
		virtual auto onDrawGui() noexcept -> void = 0;
		virtual auto onUpdate() noexcept -> void {}
	};
}