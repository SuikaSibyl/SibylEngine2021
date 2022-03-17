module;
#include <vector>
#include <unordered_map>
export module GFX.RDG.RenderGraph;
import Core.MemoryManager;
import RHI.IShader;
import RHI.IDescriptorPool;
import RHI.IFactory;
import RHI.IStorageBuffer;
import RHI.IUniformBuffer;
import RHI.ISampler;
import GFX.RDG.Common;
import GFX.RDG.ComputePassNode;
import GFX.RDG.IndirectDrawBufferNode;
import GFX.RDG.TextureBufferNode;
import GFX.RDG.ColorBufferNode;
import GFX.RDG.SamplerNode;
import GFX.RDG.RasterPassNode;

namespace SIByL::GFX::RDG
{
	struct RenderGraphBuilder;
	export class RenderGraph
	{
	public:
		auto print() noexcept -> void;

		auto getDescriptorPool() noexcept -> RHI::IDescriptorPool*;
		auto getResourceNode(NodeHandle handle) noexcept -> ResourceNode*;
		auto getIndirectDrawBufferNode(NodeHandle handle) noexcept -> IndirectDrawBufferNode*;
		auto getPassNode(NodeHandle handle) noexcept -> PassNode*;
		auto getComputePassNode(NodeHandle handle) noexcept -> ComputePassNode*;
		auto getTextureBufferNode(NodeHandle handle) noexcept -> TextureBufferNode*;
		auto getColorBufferNode(NodeHandle handle) noexcept -> ColorBufferNode*;
		auto getTextureBufferNodeFlight(NodeHandle handle, uint32_t flight) noexcept -> TextureBufferNode*;
		auto getMaxFrameInFlight() noexcept -> uint32_t { return 2; }
		auto getContainer(NodeHandle handle) noexcept -> Container*;
		auto getSamplerNode(NodeHandle handle) noexcept -> SamplerNode*;
		auto getRasterPassNode(NodeHandle handle) noexcept -> RasterPassNode*;
		auto getFramebufferContainerFlight(NodeHandle handle, uint32_t flight) noexcept -> FramebufferContainer*;
		auto getFramebufferContainer(NodeHandle handle) noexcept -> FramebufferContainer*;

		auto getUniformBufferFlight(NodeHandle handle, uint32_t const& flight) noexcept -> RHI::IUniformBuffer*;

		auto getDatumWidth() noexcept -> uint32_t { return datumWidth; }
		auto getDatumHeight() noexcept -> uint32_t { return datumHeight; }

		auto reDatum(uint32_t const& width, uint32_t const& height) noexcept -> void;

		// Node manage
		NodeRegistry registry;
		std::vector<NodeHandle> passes;
		std::vector<NodeHandle> resources;

		uint32_t storageBufferDescriptorCount = 0;
		uint32_t uniformBufferDescriptorCount = 0;
		uint32_t samplerDescriptorCount = 0;
		uint32_t storageImageDescriptorCount = 0;

		uint32_t datumWidth, datumHeight;
		RHI::IResourceFactory* factory;
		friend struct RenderGraphBuilder;
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
		auto addUniformBufferFlights(size_t size) noexcept -> NodeHandle;
		auto addStorageBuffer(size_t size) noexcept -> NodeHandle;
		auto addStorageBufferExt(RHI::IStorageBuffer* external) noexcept -> NodeHandle;
		auto addColorBufferExt(RHI::ITexture* texture, RHI::ITextureView* view, bool present =false) noexcept -> NodeHandle;
		auto addSamplerExt(RHI::ISampler* sampler) noexcept -> NodeHandle;
		auto addColorBufferFlightsExt(std::vector<RHI::ITexture*> const& textures, std::vector<RHI::ITextureView*> const& views) noexcept -> NodeHandle;
		auto addColorBufferFlightsExtPresent(std::vector<RHI::ITexture*> const& textures, std::vector<RHI::ITextureView*> const& views) noexcept -> NodeHandle;
		auto addColorBuffer(RHI::ResourceFormat format, float const& rel_width, float const& rel_height) noexcept -> NodeHandle;
		auto addIndirectDrawBuffer() noexcept -> NodeHandle;
		auto addDepthBuffer(float const& rel_width, float const& rel_height) noexcept -> NodeHandle;
		auto addFrameBufferRef(std::vector<NodeHandle> const& color_attachments, NodeHandle depth_attachment) noexcept -> NodeHandle;
		auto addFrameBufferFlightsRef(std::vector<std::pair<std::vector<NodeHandle> const&, NodeHandle>> infos) noexcept -> NodeHandle;

		auto addComputePass(RHI::IShader* shader, std::vector<NodeHandle>&& ios, uint32_t const& constant_size = 0) noexcept -> NodeHandle;
		auto addRasterPass(std::vector<NodeHandle> const& ins, uint32_t const& constant_size = 0) noexcept -> NodeHandle;

		RenderGraph& attached;

	private:
		uint32_t storageBufferCount = 0;
		uint32_t uniformBufferCount = 0;
	};

}