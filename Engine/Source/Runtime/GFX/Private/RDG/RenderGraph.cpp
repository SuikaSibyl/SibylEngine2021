module;
#include <unordered_map>
module GFX.RDG.RenderGraph;
import GFX.RDG.Common;
import GFX.RDG.IPass;
import GFX.RDG.IResource;

namespace SIByL::GFX::RDG
{
	auto RenderGraphBuilder::addTexture() noexcept -> NodeHandle
	{
		return 0;
	}

	auto RenderGraphBuilder::addUniformBuffer() noexcept -> NodeHandle
	{
		return 0;
	}

	auto RenderGraphBuilder::addStorageBuffer() noexcept -> NodeHandle
	{
		return 0;
	}

	auto RenderGraphBuilder::addComputePass() noexcept -> void
	{
		return;
	}

}