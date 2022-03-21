export module Editor.Segment;

namespace SIByL::Editor
{
	export struct Segment
	{
		virtual auto onDrawGui() noexcept -> void = 0;
	};
}