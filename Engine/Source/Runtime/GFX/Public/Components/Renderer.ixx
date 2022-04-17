module;
#include <vector>
#include <cstdint>
export module GFX.Renderer;

import GFX.Material;
import GFX.RDG.Common;

namespace SIByL::GFX
{
	export struct SubRenderer
	{
		RDG::NodeHandle pass;
		Material material;
	};

	export struct Renderer
	{
		auto hasPass(RDG::NodeHandle) noexcept -> bool;
		std::vector<SubRenderer> subRenderers;
	};

	auto Renderer::hasPass(RDG::NodeHandle pass) noexcept -> bool
	{
		for (auto& sub : subRenderers)
		{
			if (sub.pass == pass)
			{
				return true;
			}
		}
		return false;
	}

}