module;
#include <vector>
#include <filesystem>
export module RHI.IFactory;
import Core.MemoryManager;
import Core.Buffer;
import Core.Image;

import RHI.IEnum;
import RHI.GraphicContext;
import RHI.IPhysicalDevice;
import RHI.ILogicalDevice;
import RHI.ISwapChain;
import RHI.IShader;
import RHI.IFixedFunctions;
import RHI.IPipelineLayout;
import RHI.IRenderPass;
import RHI.IPipeline;
import RHI.IFramebuffer;
import RHI.ICommandPool;
import RHI.ICommandBuffer;
import RHI.ISemaphore;
import RHI.IFence;
import RHI.IBuffer;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;
import RHI.IDescriptorSetLayout;
import RHI.IUniformBuffer;
import RHI.IDescriptorPool;
import RHI.IDescriptorSet;
import RHI.IBarrier;
import RHI.IMemoryBarrier;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.ISampler;
import RHI.IStorageBuffer;

namespace SIByL
{
	namespace RHI
	{
		export struct GraphicContextDesc
		{
			API api;
		};

		export struct PhysicalDeviceDesc
		{
			IGraphicContext* context;
		};

		export struct LogicalDeviceDesc
		{
			IPhysicalDevice* physicalDevice;
		};

		export class IFactory
		{
		public:
			static auto createGraphicContext(GraphicContextDesc const& desc) noexcept -> MemScope<IGraphicContext>;
			static auto createPhysicalDevice(PhysicalDeviceDesc const& desc) noexcept -> MemScope<IPhysicalDevice>;
			static auto createLogicalDevice(LogicalDeviceDesc const& desc) noexcept -> MemScope<ILogicalDevice>;
		};

		export class IResourceFactory
		{
		public:
			IResourceFactory(ILogicalDevice* logical_device);

			auto createShaderFromBinary(Buffer const& binary, ShaderDesc const& desc) noexcept -> MemScope<IShader>;

			auto createSwapchain(SwapchainDesc const& desc) noexcept -> MemScope<ISwapChain>;
			auto createVertexLayout(BufferLayout& layout) noexcept -> MemScope<IVertexLayout>;
			auto createInputAssembly(TopologyKind const& kind) noexcept -> MemScope<IInputAssembly>;
			auto createViewportsScissors(
				unsigned int width_viewport,
				unsigned int height_viewport,
				unsigned int width_scissor,
				unsigned int height_scissor) noexcept -> MemScope<IViewportsScissors>;
			auto createViewportsScissors(Extend const& extend_viewport, Extend const& extend_scissor) noexcept -> MemScope<IViewportsScissors>;
			auto createRasterizer(RasterizerDesc const& desc) noexcept -> MemScope<IRasterizer>;
			auto createMultisampling(MultiSampleDesc const& desc) noexcept -> MemScope<IMultisampling>;
			auto createDepthStencil(DepthStencilDesc const& desc) noexcept -> MemScope<IDepthStencil>;
			auto createColorBlending(ColorBlendingDesc const& desc) noexcept -> MemScope<IColorBlending>;
			auto createDynamicState(std::vector<PipelineState> const& states) noexcept -> MemScope<IDynamicState>;
			auto createPipelineLayout(PipelineLayoutDesc const& desc) noexcept -> MemScope<IPipelineLayout>;
			auto createRenderPass(RenderPassDesc const& desc) noexcept -> MemScope<IRenderPass>;
			auto createPipeline(PipelineDesc const& desc) noexcept -> MemScope<IPipeline>;
			auto createPipeline(ComputePipelineDesc const& desc) noexcept -> MemScope<IPipeline>;
			auto createFramebuffer(FramebufferDesc const& desc) noexcept -> MemScope<IFramebuffer>;
			auto createCommandPool(CommandPoolDesc const& desc) noexcept -> MemScope<ICommandPool>;
			auto createCommandBuffer(ICommandPool* cmd_pool) noexcept -> MemScope<ICommandBuffer>;
			auto createSemaphore() noexcept -> MemScope<ISemaphore>;
			auto createFence() noexcept -> MemScope<IFence>;
			auto createVertexBuffer(Buffer* buffer) noexcept -> MemScope<IVertexBuffer>;
			auto createIndexBuffer(Buffer* buffer, uint32_t element_size) noexcept -> MemScope<IIndexBuffer>;
			auto createDescriptorSetLayout(DescriptorSetLayoutDesc const& desc) noexcept -> MemScope<IDescriptorSetLayout>;
			auto createUniformBuffer(uint32_t const& size) noexcept -> MemScope<IUniformBuffer>;
			auto createStorageBuffer(uint32_t const& size) noexcept -> MemScope<IStorageBuffer>;
			auto createStorageBuffer(Buffer* buffer) noexcept -> MemScope<IStorageBuffer>;
			auto createIndirectDrawBuffer() noexcept -> MemScope<IStorageBuffer>;
			auto createDescriptorPool(DescriptorPoolDesc const& desc) noexcept -> MemScope<IDescriptorPool>;
			auto createDescriptorSet(DescriptorSetDesc const& desc) noexcept -> MemScope<IDescriptorSet>;
			auto createMemoryBarrier(MemoryBarrierDesc const& desc) noexcept -> MemScope<IMemoryBarrier>;
			auto createBufferMemoryBarrier(BufferMemoryBarrierDesc const& desc) noexcept -> MemScope<IBufferMemoryBarrier>;
			auto createImageMemoryBarrier(ImageMemoryBarrierDesc const& desc) noexcept -> MemScope<IImageMemoryBarrier>;
			auto createBarrier(BarrierDesc const& desc) noexcept -> MemScope<IBarrier>;
			auto createBufferImageCopy(BufferImageCopyDesc const& desc) noexcept -> MemScope<IBufferImageCopy>;
			auto createTexture(Image* image) noexcept -> MemScope<ITexture>;
			auto createTexture(TextureDesc const&) noexcept -> MemScope<ITexture>;
			auto createTextureView(ITexture* texture, ImageUsageFlags extra_usages = 0) noexcept -> MemScope<ITextureView>;
			auto createSampler(SamplerDesc const&) noexcept -> MemScope<ISampler>;

			auto createShaderFromBinaryFile(std::filesystem::path path, ShaderDesc const& desc) noexcept -> MemScope<IShader>;

		private:
			API api;
			IGraphicContext* graphicContext;
			IPhysicalDevice* physicalDevice;
			ILogicalDevice* logicalDevice;
		};
	}
}