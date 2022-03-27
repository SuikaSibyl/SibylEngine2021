module;
#include <vector>
#include <unordered_map>
#include <string_view>
export module GFX.RDG.RenderGraph;
import Core.MemoryManager;
import RHI.IShader;
import RHI.IDescriptorPool;
import RHI.IFactory;
import RHI.IStorageBuffer;
import RHI.IUniformBuffer;
import RHI.ISampler;
import RHI.ICommandBuffer;
import RHI.ICommandBuffer;
import GFX.RDG.Common;
import GFX.RDG.ComputePassNode;
import GFX.RDG.IndirectDrawBufferNode;
import GFX.RDG.TextureBufferNode;
import GFX.RDG.ColorBufferNode;
import GFX.RDG.SamplerNode;
import GFX.RDG.RasterPassNode;
import GFX.RDG.MultiDispatchScope;

namespace SIByL::GFX::RDG
{
	struct RenderGraphBuilder;

	// Life of a Render Graph
	// 
	// ╭── Render Graph Begin
	// │ 
	// │ ╭── Render Graph Builder Begin
	// │ ├────	Register Resources Nodes
	// │ ├────	Register Passes Nodes
	// │ ╰── Render Graph Builder End
	// │ 
	// │ ╭── Build Begin
	// │ │ ╭── Unvirtualize Begin
	// │ │ ├────  Unvirtualize Resources Nodes
	// │ │ ├────  Unvirtualize Passes Nodes
	// │ │ ╰── Unvirtualize Begin
	// │ │ ╭── Compile Pipeline Begin
	// │ │ ├────  Consume Hisotry Insertion
	// │ │ ├────  Barrier Creation
	// │ │ ╰── Compile Pipeline Begin
	// │ ╰── Build End
	// │ 
	// │ ╭── Run Begin
	// │ ╰── Run End
	// │
	// │ ~ OPTIONAL ~
	// │ ╭── Render Graph Builder(Editor) Begin
	// │ ├────	Register / Modify Resources Nodes
	// │ ├────	Register / Modify Passes Nodes
	// │ │
	// │ │ ╭── Unvirtualize Begin
	// │ │ ├────  Unvirtualize Resources Nodes (Only Dirty)
	// │ │ ├────  Unvirtualize Passes Nodes (Only Dirty)
	// │ │ ╰── Unvirtualize End
	// │ │ 
	// │ │ ╭── Compile Pipeline Begin
	// │ │ ├────  Consume Hisotry Insertion
	// │ │ ├────  Barrier Creation
	// │ │ ╰── Compile Pipeline Begin
	// │ │ 
	// │ ╰── Render Graph Builder(Editor) End
	// │ 
	// ╰── Render Graph End
	//

	export class RenderGraph
	{
	public:
		auto print() noexcept -> void;
		auto tag(NodeHandle handle, std::string_view tag) noexcept -> void;

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
		auto getPassScope(NodeHandle handle) noexcept -> PassScope*;
		auto getMultiDispatchScope(NodeHandle handle) noexcept -> MultiDispatchScope*;

		auto getUniformBufferFlight(NodeHandle handle, uint32_t const& flight) noexcept -> RHI::IUniformBuffer*;

		auto getDatumWidth() noexcept -> uint32_t { return datumWidth; }
		auto getDatumHeight() noexcept -> uint32_t { return datumHeight; }

		auto reDatum(uint32_t const& width, uint32_t const& height) noexcept -> void;
		auto recordCommands(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void;

		// Node manage
		NodeRegistry registry;
		std::vector<NodeHandle> passes;
		std::vector<NodeHandle> passesBackPool;
		std::vector<NodeHandle> resources;

		uint32_t storageBufferDescriptorCount = 0;
		uint32_t uniformBufferDescriptorCount = 0;
		uint32_t samplerDescriptorCount = 0;
		uint32_t storageImageDescriptorCount = 0;

		BarrierPool barrierPool;
		uint32_t datumWidth, datumHeight;
		RHI::IResourceFactory* factory;
		friend struct RenderGraphBuilder;
		MemScope<RHI::IDescriptorPool> descriptorPool;
	};

	export struct RenderGraphBuilder
	{
		RenderGraphBuilder(RenderGraph& attached) :attached(attached) {}

		// life
		auto build(RHI::IResourceFactory* factory, uint32_t const& width, uint32_t const& height) noexcept -> void;

		// add resource nodes
		auto addTexture() noexcept -> NodeHandle;
		auto addUniformBuffer(size_t size) noexcept -> NodeHandle;
		auto addUniformBufferFlights(size_t size) noexcept -> NodeHandle;
		auto addStorageBuffer(size_t size, std::string_view name) noexcept -> NodeHandle;
		auto addStorageBufferExt(RHI::IStorageBuffer* external, std::string_view name) noexcept -> NodeHandle;
		auto addColorBufferExt(RHI::ITexture* texture, RHI::ITextureView* view, std::string_view name, bool present =false) noexcept -> NodeHandle;
		auto addSamplerExt(RHI::ISampler* sampler) noexcept -> NodeHandle;
		auto addColorBufferFlightsExt(std::vector<RHI::ITexture*> const& textures, std::vector<RHI::ITextureView*> const& views) noexcept -> NodeHandle;
		auto addColorBufferFlightsExtPresent(std::vector<RHI::ITexture*> const& textures, std::vector<RHI::ITextureView*> const& views) noexcept -> NodeHandle;
		auto addColorBuffer(RHI::ResourceFormat format, float const& rel_width, float const& rel_height, std::string_view name) noexcept -> NodeHandle;
		auto addIndirectDrawBuffer(std::string_view name) noexcept -> NodeHandle;
		auto addDepthBuffer(float const& rel_width, float const& rel_height) noexcept -> NodeHandle;
		auto addFrameBufferRef(std::vector<NodeHandle> const& color_attachments, NodeHandle depth_attachment) noexcept -> NodeHandle;
		auto addFrameBufferFlightsRef(std::vector<std::pair<std::vector<NodeHandle> const&, NodeHandle>> infos) noexcept -> NodeHandle;
		auto beginMultiDispatchScope(std::string_view name) noexcept -> NodeHandle;
		auto endScope() noexcept -> NodeHandle;

		auto addComputePass(RHI::IShader* shader, std::vector<NodeHandle>&& ios, std::string_view name, uint32_t const& constant_size = 0) noexcept -> NodeHandle;
		auto addComputePassBackPool(RHI::IShader* shader, std::vector<NodeHandle>&& ios, std::string_view name, uint32_t const& constant_size = 0) noexcept -> NodeHandle;
		auto addRasterPass(std::vector<NodeHandle> const& ins, uint32_t const& constant_size = 0) noexcept -> NodeHandle;
		auto addRasterPassBackPool(std::vector<NodeHandle> const& ins, uint32_t const& constant_size = 0) noexcept -> NodeHandle;

		RenderGraph& attached;
		std::vector<NodeHandle> scopeStack;
	};
}