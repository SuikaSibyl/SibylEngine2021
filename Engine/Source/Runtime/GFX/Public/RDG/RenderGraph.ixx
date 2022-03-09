module;
#include <unordered_map>
export module GFX.RDG.RenderGraph;
import Core.MemoryManager;
import RHI.IShader;
import RHI.IDescriptorPool;
import RHI.IFactory;
import GFX.RDG.Common;
import GFX.RDG.PassNode;
import GFX.RDG.ResourceNode;
import GFX.RDG.ComputePassNode;

namespace SIByL::GFX::RDG
{
	struct RenderGraphBuilder;
	export class RenderGraph
	{
	public:
		auto getDescriptorPool() noexcept -> RHI::IDescriptorPool*;
		auto getResourceNode(NodeHandle handle) noexcept -> ResourceNode*;
		auto getPassNode(NodeHandle handle) noexcept -> PassNode*;
		auto getComputePassNode(NodeHandle handle) noexcept -> ComputePassNode*;
		auto getMaxFrameInFlight() noexcept -> uint32_t { return 2; }

	private:
		friend struct RenderGraphBuilder;
		std::unordered_map<NodeHandle, MemScope<PassNode>> passes;
		std::unordered_map<NodeHandle, MemScope<ResourceNode>> resources;
		MemScope<RHI::IDescriptorPool> descriptorPool;
	};

	export struct RenderGraphBuilder
	{
		RenderGraphBuilder(RenderGraph& attached) :attached(attached) {}

		// life
		auto build(RHI::IResourceFactory* factory) noexcept -> void;

		// add resource nodes
		auto addTexture() noexcept -> NodeHandle;
		auto addUniformBuffer(size_t size) noexcept -> NodeHandle;
		auto addStorageBuffer(size_t size) noexcept -> NodeHandle;

		auto addComputePass(RHI::IShader* shader, std::vector<NodeHandle>&& ios, uint32_t const& constant_size = 0) noexcept -> NodeHandle;

		RenderGraph& attached;

	private:
		uint32_t storageBufferCount = 0;
		uint32_t uniformBufferCount = 0;
	};

}