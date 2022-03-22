module;
#include <utility>
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include <string_view>
module GFX.RDG.Common;
import Core.BitFlag;
import Core.Log;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.IFactory;
import RHI.ICommandBuffer;
import RHI.IRenderPass;
import RHI.IFramebuffer;
import ECS.UID;
import GFX.RDG.RenderGraph;
import GFX.RDG.TextureBufferNode;

namespace SIByL::GFX::RDG
{
	auto Node::onPrint() noexcept -> void
	{
		std::string type_str = hasBit(attributes, NodeAttrbutesFlagBits::RESOURCE) ? 
			"Resource" : 
			"Pass    ";
		std::string detailed_type_str;

		switch (type)
		{
		case NodeDetailedType::NONE: 			detailed_type_str = "NONE          "; break;
		case NodeDetailedType::STORAGE_BUFFER: 	detailed_type_str = "STORAGE_BUFFER"; break;
		case NodeDetailedType::UNIFORM_BUFFER: 	detailed_type_str = "UNIFORM_BUFFER"; break;
		case NodeDetailedType::FRAME_BUFFER: 	detailed_type_str = "FRAME_BUFFER  "; break;
		case NodeDetailedType::SAMPLER: 		detailed_type_str = "SAMPLER       "; break;
		case NodeDetailedType::COLOR_TEXTURE:   detailed_type_str = "COLOR_TEXTURE "; break;
		case NodeDetailedType::DEPTH_TEXTURE: 	detailed_type_str = "DEPTH_TEXTURE "; break;
		case NodeDetailedType::RASTER_PASS: 	detailed_type_str = "RASTER PASS   "; break;
		case NodeDetailedType::COMPUTE_PASS: 	detailed_type_str = "COMPUTE PASS  "; break;
		case NodeDetailedType::BLIT_PASS: 		detailed_type_str = "BLIT PASS     "; break;
		default:								detailed_type_str = "ERROR         "; break;
		}

		std::string is_flight_str = hasBit(attributes, NodeAttrbutesFlagBits::FLIGHT) ?
			"Flights" : 
			"Atom   ";
		//std::string_view tag;
		if (hasBit(attributes, NodeAttrbutesFlagBits::RESOURCE))
		{
			std::string cosume_history = {};
			ResourceNode* resource_node = (ResourceNode*)this;
			for (int i = 0; i < resource_node->consumeHistory.size(); i++)
			{
				std::string one_history = getString(resource_node->consumeHistory[i].kind);
				cosume_history.append(one_history);
			}
			SE_CORE_INFO("│ NODE │ {0} │ {1} │ {2} | {3}", detailed_type_str, is_flight_str, (tag != std::string_view{}) ? tag.data() : "NAMELESS", cosume_history);
		}
		else
		{
			SE_CORE_INFO("│ NODE │ {0} │ {1} │ {2} ", detailed_type_str, is_flight_str, (tag != std::string_view{}) ? tag.data() : "NAMELESS");
		}
	}

	auto BarrerPool::registBarrier(MemScope<RHI::IBarrier>&& barrier) noexcept -> NodeHandle
	{
		uint64_t uid = ECS::UniqueID::RequestUniqueID();
		barriers[uid] = std::move(barrier);
		return uid;
	}

	auto BarrerPool::getBarrier(NodeHandle handle) noexcept -> RHI::IBarrier*
	{
		return barriers[handle].get();
	}

	auto FramebufferContainer::getWidth() noexcept -> uint32_t 
	{ 
		if (colorAttachCount > 0 || depthAttachCount > 0)
		{
			return ((TextureBufferNode*)registry->getNode(handles[handles.size()-1]))->getTexture()->getDescription().width;
		}
		else
		{
			SE_CORE_ERROR("RDG :: Framebuffer creation without either color attachments or depth attachments");
			return 0;
		}
	}

	auto FramebufferContainer::getHeight() noexcept -> uint32_t 
	{ 
		if (colorAttachCount > 0 || depthAttachCount > 0)
		{
			return ((TextureBufferNode*)registry->getNode(handles[handles.size() - 1]))->getTexture()->getDescription().height;
		}
		else
		{
			SE_CORE_ERROR("RDG :: Framebuffer creation without either color attachments or depth attachments");
			return 0;
		}
	}
	
	auto FramebufferContainer::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;
		
		RHI::RenderPassDesc renderpass_desc;
		renderpass_desc.colorAttachments.resize(colorAttachCount);
		renderpass_desc.depthstencialAttachments.resize(depthAttachCount);
		for (int i = 0; i < colorAttachCount; i++)
		{
			renderpass_desc.colorAttachments[i] =
			{
				RHI::SampleCount::COUNT_1_BIT,
				rg->getColorBufferNode(handles[i])->format,
				RHI::AttachmentLoadOp::CLEAR,
				RHI::AttachmentStoreOp::STORE,
				RHI::AttachmentLoadOp::DONT_CARE,
				RHI::AttachmentStoreOp::DONT_CARE,
				RHI::ImageLayout::UNDEFINED,
				hasBit(rg->getColorBufferNode(handles[i])->attributes, NodeAttrbutesFlagBits::PRESENT) ? RHI::ImageLayout::PRESENT_SRC : RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
				{0,0,0,1}
			};
		}
		for (int i = 0; i < depthAttachCount; i++)
		{
			renderpass_desc.depthstencialAttachments[i] =
			{
				RHI::SampleCount::COUNT_1_BIT,
				RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT,
				RHI::AttachmentLoadOp::CLEAR,
				RHI::AttachmentStoreOp::DONT_CARE,
				RHI::AttachmentLoadOp::DONT_CARE,
				RHI::AttachmentStoreOp::DONT_CARE,
				RHI::ImageLayout::UNDEFINED,
				RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA,
				{ 1,0 }
			};
		}
		renderPass = factory->createRenderPass(renderpass_desc);
		onReDatum(graph, factory);
	}

	auto FramebufferContainer::onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;
		std::vector<RHI::ITextureView*> attachments(handles.size());
		for (int i = 0; i < handles.size(); i++)
		{
			attachments[i] = rg->getTextureBufferNode(handles[i])->getTextureView();
		}
		RHI::FramebufferDesc framebuffer_desc =
		{
			getWidth(),
			getHeight(),
			renderPass.get(),
			attachments,
		};
		framebuffer = factory->createFramebuffer(framebuffer_desc);
	}
}