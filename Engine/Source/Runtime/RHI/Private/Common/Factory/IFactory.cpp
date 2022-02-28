module;
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <string_view>
#include <filesystem>
module RHI.IFactory;
import Core.SObject;
import Core.SPointer;
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

import RHI.GraphicContext.VK;
import RHI.IPhysicalDevice.VK;
import RHI.ILogicalDevice.VK;
import RHI.ISwapChain.VK;
import RHI.IShader.VK;
import RHI.IFixedFunctions.VK;
import RHI.IPipelineLayout.VK;
import RHI.IRenderPass.VK;
import RHI.IPipeline.VK;
import RHI.IFramebuffer.VK;
import RHI.ICommandPool.VK;
import RHI.ICommandBuffer.VK;
import RHI.ISemaphore.VK;
import RHI.IFence.VK;
import RHI.IVertexBuffer.VK;
import RHI.IIndexBuffer.VK;
import RHI.IDescriptorSetLayout.VK;
import RHI.IUniformBuffer.VK;
import RHI.IDescriptorPool.VK;
import RHI.IDescriptorSet.VK;
import RHI.IBarrier.VK;
import RHI.IMemoryBarrier.VK;
import RHI.ITexture.VK;
import RHI.ITextureView.VK;
import RHI.ISampler.VK;

namespace SIByL::RHI
{
	auto IFactory::createGraphicContext(GraphicContextDesc const& desc) noexcept -> IGraphicContext*
	{
		IGraphicContext* res = nullptr;
		switch (desc.api)
		{
		case API::VULKAN:
			res = SNew<IGraphicContextVK>();
			break;
		case API::DX12:
			break;
		default:
			break;
		}
		if (res != nullptr)
		{
			res->setAPI(desc.api);
		}
		return res;
	}

	auto IFactory::createPhysicalDevice(PhysicalDeviceDesc const& desc) noexcept -> IPhysicalDevice*
	{
		IPhysicalDevice* res = nullptr;
		switch (desc.context->getAPI())
		{
		case API::VULKAN:
			res = SNew<IPhysicalDeviceVK>(desc.context);
			break;
		case API::DX12:
			break;
		default:
			break;
		}
		return res;
	}

	auto IFactory::createLogicalDevice(LogicalDeviceDesc const& desc) noexcept -> ILogicalDevice*
	{
		IPhysicalDevice* physical_device = desc.physicalDevice;
		ILogicalDevice* res = nullptr;
		switch (physical_device->getGraphicContext()->getAPI())
		{
		case API::VULKAN:
			res = SNew<ILogicalDeviceVK>((IPhysicalDeviceVK*)physical_device);
			break;
		case API::DX12:
			break;
		default:
			break;
		}
		return res;
	}

	IResourceFactory::IResourceFactory(ILogicalDevice* logical_device)
	{
		logicalDevice = logical_device;
		physicalDevice = logicalDevice->getPhysicalDevice();
		graphicContext = physicalDevice->getGraphicContext();
		api = graphicContext->getAPI();
	}

	auto IResourceFactory::createSwapchain(SwapchainDesc const& desc) noexcept -> MemScope<ISwapChain>
	{
		MemScope<ISwapChain> sc = nullptr;

		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{			
			MemScope<ISwapChainVK> sc_vk = MemNew<ISwapChainVK>(desc, static_cast<RHI::ILogicalDeviceVK*>(logicalDevice));
			sc = MemCast<ISwapChain>(sc_vk);
		}
		break;
		default:
			break;
		}

		return sc;
	}

	auto IResourceFactory::createShaderFromBinary(Buffer const& binary, ShaderDesc const& desc) noexcept -> MemScope<IShader>
	{
		MemScope<IShader> shader = nullptr;

		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<RHI::IShaderVK> shader_vk = MemNew<RHI::IShaderVK>(static_cast<RHI::ILogicalDeviceVK*>(logicalDevice));
			shader_vk->injectDesc(desc);
			shader_vk->createShaderModule(binary.getData(), binary.getSize());
			shader_vk->createVkShaderStageCreateInfo();
			shader = std::move(MemCast<RHI::IShader>(shader_vk));
		}
			break;
		default:
			break;
		}

		return shader;
	}

	auto IResourceFactory::createVertexLayout(BufferLayout& _layout) noexcept -> MemScope<IVertexLayout>
	{
		MemScope<IVertexLayout> layout = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IVertexLayoutVK> layout_vk = MemNew<IVertexLayoutVK>(_layout);
			layout = MemCast<IVertexLayout>(layout_vk);
		}
		break;
		default:
			break;
		}
		return layout;
	}

	auto IResourceFactory::createInputAssembly(TopologyKind const& kind) noexcept -> MemScope<IInputAssembly>
	{
		MemScope<IInputAssembly> assembly = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IInputAssemblyVK> assembly_vk = MemNew<IInputAssemblyVK>(kind);
			assembly = MemCast<IInputAssembly>(assembly_vk);
		}
		break;
		default:
			break;
		}
		return assembly;
	}

	auto IResourceFactory::createViewportsScissors(
		unsigned int width_viewport,
		unsigned int height_viewport,
		unsigned int width_scissor,
		unsigned int height_scissor) noexcept -> MemScope<IViewportsScissors>
	{
		MemScope<IViewportsScissors> vs = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IViewportsScissorsVK> vs_vk = MemNew<IViewportsScissorsVK>(
				width_viewport,
				height_viewport,
				width_scissor,
				height_scissor);
			vs = MemCast<IViewportsScissors>(vs_vk);
		}
		break;
		default:
			break;
		}
		return vs;
	}

	auto IResourceFactory::createViewportsScissors(
		Extend const& extend_viewport, 
		Extend const& extend_scissor) noexcept -> MemScope<IViewportsScissors>
	{
		MemScope<IViewportsScissors> vs = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IViewportsScissorsVK> vs_vk = MemNew<IViewportsScissorsVK>(
				extend_viewport.width,
				extend_viewport.height,
				extend_scissor.width,
				extend_scissor.height);
			vs = MemCast<IViewportsScissors>(vs_vk);
		}
		break;
		default:
			break;
		}
		return vs;
	}

	auto IResourceFactory::createRasterizer(RasterizerDesc const& desc) noexcept -> MemScope<IRasterizer>
	{
		MemScope<IRasterizer> rasterizer = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IRasterizerVK> rasterizer_vk = MemNew<IRasterizerVK>(desc);
			rasterizer = MemCast<IRasterizer>(rasterizer_vk);
		}
		break;
		default:
			break;
		}
		return rasterizer;
	}

	auto IResourceFactory::createMultisampling(MultiSampleDesc const& desc) noexcept -> MemScope<IMultisampling>
	{
		MemScope<IMultisampling> ms = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IMultisamplingVK> ms_vk = MemNew<IMultisamplingVK>(desc);
			ms = MemCast<IMultisampling>(ms_vk);
		}
		break;
		default:
			break;
		}
		return ms;
	}

	auto IResourceFactory::createDepthStencil(DepthStencilDesc const& desc) noexcept -> MemScope<IDepthStencil>
	{
		MemScope<IDepthStencil> ds = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IDepthStencilVK> ds_vk = MemNew<IDepthStencilVK>(desc);
			ds = MemCast<IDepthStencil>(ds_vk);
		}
		break;
		default:
			break;
		}
		return ds;
	}

	auto IResourceFactory::createColorBlending(ColorBlendingDesc const& desc) noexcept -> MemScope<IColorBlending>
	{
		MemScope<IColorBlending> cb = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IColorBlendingVK> cb_vk = MemNew<IColorBlendingVK>(desc);
			cb = MemCast<IColorBlending>(cb_vk);
		}
			break;
		default:
			break;
		}
		return cb;
	}

	auto IResourceFactory::createDynamicState(std::vector<PipelineState> const& states) noexcept -> MemScope<IDynamicState>
	{
		MemScope<IDynamicState> ds = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IDynamicStateVK> ds_vk = MemNew<IDynamicStateVK>(states);
			ds = MemCast<IDynamicState>(ds_vk);
		}
		break;
		default:
			break;
		}
		return ds;
	}

	auto IResourceFactory::createPipelineLayout(PipelineLayoutDesc const& desc) noexcept -> MemScope<IPipelineLayout>
	{
		MemScope<IPipelineLayout> pl = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IPipelineLayoutVK> pl_vk = MemNew<IPipelineLayoutVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			pl = MemCast<IPipelineLayout>(pl_vk);
		}
		break;
		default:
			break;
		}
		return pl;
	}

	auto IResourceFactory::createRenderPass(RenderPassDesc const& desc) noexcept -> MemScope<IRenderPass>
	{
		MemScope<IRenderPass> rp = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IRenderPassVK> rp_vk = MemNew<IRenderPassVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			rp = MemCast<IRenderPass>(rp_vk);
		}
		break;
		default:
			break;
		}
		return rp;
	}

	auto IResourceFactory::createPipeline(PipelineDesc const& desc) noexcept -> MemScope<IPipeline>
	{
		MemScope<IPipeline> rp = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IPipelineVK> rp_vk = MemNew<IPipelineVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			rp = MemCast<IPipeline>(rp_vk);
		}
		break;
		default:
			break;
		}
		return rp;
	}

	auto IResourceFactory::createFramebuffer(FramebufferDesc const& desc) noexcept -> MemScope<IFramebuffer>
	{
		MemScope<IFramebuffer> fb = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IFramebufferVK> fb_vk = MemNew<IFramebufferVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			fb = MemCast<IFramebuffer>(fb_vk);
		}
		break;
		default:
			break;
		}
		return fb;
	}

	auto IResourceFactory::createCommandPool(CommandPoolDesc const& desc) noexcept -> MemScope<ICommandPool>
	{
		MemScope<ICommandPool> cp = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<ICommandPoolVK> cp_vk = MemNew<ICommandPoolVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			cp = MemCast<ICommandPool>(cp_vk);
		}
		break;
		default:
			break;
		}
		return cp;
	}

	auto IResourceFactory::createCommandBuffer(ICommandPool* cmd_pool) noexcept -> MemScope<ICommandBuffer>
	{
		MemScope<ICommandBuffer> cb = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<ICommandBufferVK> cb_vk = MemNew<ICommandBufferVK>((ICommandPoolVK*)cmd_pool, (ILogicalDeviceVK*)logicalDevice);
			cb = MemCast<ICommandBuffer>(cb_vk);
		}
		break;
		default:
			break;
		}
		return cb;
	}

	auto IResourceFactory::createSemaphore() noexcept -> MemScope<ISemaphore>
	{
		MemScope<ISemaphore> semaphore = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<ISemaphoreVK> semaphore_vk = MemNew<ISemaphoreVK>((ILogicalDeviceVK*)logicalDevice);
			semaphore = MemCast<ISemaphore>(semaphore_vk);
		}
		break;
		default:
			break;
		}
		return semaphore;
	}

	auto IResourceFactory::createFence() noexcept -> MemScope<IFence>
	{
		MemScope<IFence> fence = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IFenceVK> fence_vk = MemNew<IFenceVK>((ILogicalDeviceVK*)logicalDevice);
			fence = MemCast<IFence>(fence_vk);
		}
		break;
		default:
			break;
		}
		return fence;
	}

	auto IResourceFactory::createVertexBuffer(Buffer* buffer) noexcept -> MemScope<IVertexBuffer>
	{
		MemScope<IVertexBuffer> vb = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IVertexBufferVK> vb_vk = MemNew<IVertexBufferVK>(buffer, (ILogicalDeviceVK*)logicalDevice);
			vb = MemCast<IVertexBuffer>(vb_vk);
		}
		break;
		default:
			break;
		}
		return vb;
	}

	auto IResourceFactory::createIndexBuffer(Buffer* buffer, uint32_t element_size) noexcept -> MemScope<IIndexBuffer>
	{
		MemScope<IIndexBuffer> ib = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IIndexBufferVK> ib_vk = MemNew<IIndexBufferVK>(buffer, element_size, (ILogicalDeviceVK*)logicalDevice);
			ib = MemCast<IIndexBuffer>(ib_vk);
		}
		break;
		default:
			break;
		}
		return ib;
	}

	auto IResourceFactory::createDescriptorSetLayout(DescriptorSetLayoutDesc const& desc) noexcept -> MemScope<IDescriptorSetLayout>
	{
		MemScope<IDescriptorSetLayout> dl = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IDescriptorSetLayoutVK> dl_vk = MemNew<IDescriptorSetLayoutVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			dl = MemCast<IDescriptorSetLayout>(dl_vk);
		}
		break;
		default:
			break;
		}
		return dl;
	}

	auto IResourceFactory::createUniformBuffer(uint32_t const& size) noexcept -> MemScope<IUniformBuffer>
	{
		MemScope<IUniformBuffer> ub = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IUniformBufferVK> ub_vk = MemNew<IUniformBufferVK>(size, (ILogicalDeviceVK*)logicalDevice);
			ub = MemCast<IUniformBuffer>(ub_vk);
		}
		break;
		default:
			break;
		}
		return ub;
	}

	auto IResourceFactory::createDescriptorPool(DescriptorPoolDesc const& desc) noexcept -> MemScope<IDescriptorPool>
	{
		MemScope<IDescriptorPool> dp = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IDescriptorPoolVK> dp_vk = MemNew<IDescriptorPoolVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			dp = MemCast<IDescriptorPool>(dp_vk);
		}
		break;
		default:
			break;
		}
		return dp;
	}

	auto IResourceFactory::createDescriptorSet(DescriptorSetDesc const& desc) noexcept -> MemScope<IDescriptorSet>
	{
		MemScope<IDescriptorSet> ds = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IDescriptorSetVK> ds_vk = MemNew<IDescriptorSetVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			ds = MemCast<IDescriptorSet>(ds_vk);
		}
		break;
		default:
			break;
		}
		return ds;
	}

	auto IResourceFactory::createImageMemoryBarrier(ImageMemoryBarrierDesc const& desc) noexcept -> MemScope<IImageMemoryBarrier>
	{
		MemScope<IImageMemoryBarrier> imb = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IImageMemoryBarrierVK> imb_vk = MemNew<IImageMemoryBarrierVK>(desc);
			imb = MemCast<IImageMemoryBarrier>(imb_vk);
		}
		break;
		default:
			break;
		}
		return imb;
	}

	auto IResourceFactory::createBarrier(BarrierDesc const& desc) noexcept -> MemScope<IBarrier>
	{
		MemScope<IBarrier> barrier = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IBarrierVK> barrier_vk = MemNew<IBarrierVK>(desc);
			barrier = MemCast<IBarrier>(barrier_vk);
		}
		break;
		default:
			break;
		}
		return barrier;
	}

	auto IResourceFactory::createBufferImageCopy(BufferImageCopyDesc const& desc) noexcept -> MemScope<IBufferImageCopy>
	{
		MemScope<IBufferImageCopy> bic = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IBufferImageCopyVK> bic_vk = MemNew<IBufferImageCopyVK>(desc);
			bic = MemCast<IBufferImageCopy>(bic_vk);
		}
		break;
		default:
			break;
		}
		return bic;
	}

	auto IResourceFactory::createTexture(Image* image) noexcept -> MemScope<ITexture>
	{
		MemScope<ITexture> tx = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<ITextureVK> tx_vk = MemNew<ITextureVK>(image, (ILogicalDeviceVK*)logicalDevice);
			tx = MemCast<ITexture>(tx_vk);
		}
		break;
		default:
			break;
		}
		return tx;
	}

	auto IResourceFactory::createTexture(TextureDesc const& desc) noexcept -> MemScope<ITexture>
	{
		MemScope<ITexture> tx = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<ITextureVK> tx_vk = MemNew<ITextureVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			tx = MemCast<ITexture>(tx_vk);
		}
		break;
		default:
			break;
		}
		return tx;
	}

	auto IResourceFactory::createTextureView(ITexture* texture) noexcept -> MemScope<ITextureView>
	{
		MemScope<ITextureView> tv = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<ITextureViewVK> tv_vk = MemNew<ITextureViewVK>(texture, (ILogicalDeviceVK*)logicalDevice);
			tv = MemCast<ITextureView>(tv_vk);
		}
		break;
		default:
			break;
		}
		return tv;
	}

	auto IResourceFactory::createSampler(SamplerDesc const& desc) noexcept -> MemScope<ISampler>
	{
		MemScope<ISampler> sampler = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<ISamplerVK> sampler_vk = MemNew<ISamplerVK>(desc, (ILogicalDeviceVK*)logicalDevice);
			sampler = MemCast<ISampler>(sampler_vk);
		}
		break;
		default:
			break;
		}
		return sampler;
	}

}