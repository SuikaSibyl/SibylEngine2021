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
import GFX.RDG.IndirectDrawBufferNode;
import GFX.RDG.TextureBufferNode;

namespace SIByL::GFX::RDG
{
	struct RenderGraphBuilder;
	export class RenderGraph
	{
	public:
		auto getDescriptorPool() noexcept -> RHI::IDescriptorPool*;
		auto getResourceNode(NodeHandle handle) noexcept -> ResourceNode*;
		auto getIndirectDrawBufferNode(NodeHandle handle) noexcept -> IndirectDrawBufferNode*;
		auto getPassNode(NodeHandle handle) noexcept -> PassNode*;
		auto getComputePassNode(NodeHandle handle) noexcept -> ComputePassNode*;
		auto getTextureBufferNode(NodeHandle handle) noexcept -> TextureBufferNode*;
		auto getMaxFrameInFlight() noexcept -> uint32_t { return 2; }

		auto getDatumWidth() noexcept -> uint32_t { return datumWidth; }
		auto getDatumHeight() noexcept -> uint32_t { return datumHeight; }

		auto reDatum(uint32_t const& width, uint32_t const& height) noexcept -> void;

	private:
		uint32_t datumWidth, datumHeight;
		RHI::IResourceFactory* factory;
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
		auto addUniformBufferFlights(size_t size) noexcept -> Container;
		auto addStorageBuffer(size_t size) noexcept -> NodeHandle;
		auto addIndirectDrawBuffer() noexcept -> NodeHandle;
		auto addDepthBuffer(float const& rel_width, float const& rel_height) noexcept -> NodeHandle;

		auto addComputePass(RHI::IShader* shader, std::vector<NodeHandle>&& ios, uint32_t const& constant_size = 0) noexcept -> NodeHandle;

		RenderGraph& attached;

	private:
		uint32_t storageBufferCount = 0;
		uint32_t uniformBufferCount = 0;
	};

}