module;
#include <unordered_map>
export module GFX.RDG.RenderGraph;
import GFX.RDG.Common;
import GFX.RDG.IPass;
import GFX.RDG.IResource;

namespace SIByL::GFX::RDG
{
	export class RenderGraph
	{
	public:

	private:
		std::unordered_map<NodeHandle, PassNode> passes;
		std::unordered_map<NodeHandle, IResource> resources;

	};

	export struct RenderGraphBuilder
	{
		// life
		auto execute() noexcept -> void;

		// add resource nodes
		auto addTexture() noexcept -> NodeHandle;
		auto addUniformBuffer() noexcept -> NodeHandle;
		auto addStorageBuffer() noexcept -> NodeHandle;

		auto addComputePass() noexcept -> void;
	};

}