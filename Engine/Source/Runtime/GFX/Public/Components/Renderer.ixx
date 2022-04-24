module;
#include <vector>
#include <cstdint>
#include <string>
export module GFX.Renderer;

import GFX.Material;
import GFX.RDG.Common;

namespace SIByL::GFX
{
	export struct SubRenderer
	{
		std::string passName;
		std::string pipelineName;
		std::string materialName;
	};

	export struct Renderer
	{
		auto hasPass(RDG::NodeHandle) noexcept -> bool;
		std::vector<SubRenderer> subRenderers;
	};
}