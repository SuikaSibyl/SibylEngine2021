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

import RHI.GraphicContext.VK;
import RHI.IPhysicalDevice.VK;
import RHI.ILogicalDevice.VK;
import RHI.ISwapChain.VK;
import RHI.IShader.VK;
import RHI.IFixedFunctions.VK;
import RHI.IPipelineLayout.VK;
import RHI.IRenderPass.VK;
import RHI.IPipeline.VK;

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

	auto IFactory::createSwapchain(SwapchainDesc const& desc) noexcept -> ISwapChain*
	{
		ISwapChain* swapchain = nullptr;
		switch (desc.logicalDevice->getPhysicalDevice()->getGraphicContext()->getAPI())
		{
		case API::VULKAN:
			swapchain = SNew<ISwapChainVK>((ILogicalDeviceVK*)desc.logicalDevice);
			break;
		case API::DX12:
			break;
		default:
			break;
		}
		return swapchain;
	}

	IResourceFactory::IResourceFactory(ILogicalDevice* logical_device)
	{
		logicalDevice = logical_device;
		physicalDevice = logicalDevice->getPhysicalDevice();
		graphicContext = physicalDevice->getGraphicContext();
		api = graphicContext->getAPI();
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

	auto IResourceFactory::createVertexLayout() noexcept -> MemScope<IVertexLayout>
	{
		MemScope<IVertexLayout> layout = nullptr;
		switch (api)
		{
		case SIByL::RHI::API::DX12:
			break;
		case SIByL::RHI::API::VULKAN:
		{
			MemScope<IVertexLayoutVK> layout_vk = MemNew<IVertexLayoutVK>();
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
}