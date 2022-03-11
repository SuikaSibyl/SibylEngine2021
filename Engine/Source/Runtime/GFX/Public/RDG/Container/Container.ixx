module;
#include <cstdint>
#include <vector>
#include <string>
export module GFX.RDG.Container;
import GFX.RDG.Common;

namespace SIByL::GFX::RDG
{
	export struct FrameFlightContainer :public Container
	{

	};

	export struct FramebufferContainer :public Container
	{
		auto getWidth() noexcept -> uint32_t;
		auto getHeight() noexcept -> uint32_t;

		uint32_t width, height;
	};
}