module;
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IRenderPass.VK;
import Core.Log;
import Core.Color;
import RHI.IRenderPass;
import RHI.ILogicalDevice.VK;
import RHI.IEnum;
import RHI.IEnum.VK;
import RHI.IResource.VK;

namespace SIByL
{
	namespace RHI
	{
		auto getVkAttachmentLoadOp(AttachmentLoadOp op) noexcept -> VkAttachmentLoadOp;
		auto getVkAttachmentStorep(AttachmentStoreOp op) noexcept -> VkAttachmentStoreOp;

		auto createVkAttachmentDescriotion(AttachmentDesc const& desc) noexcept -> VkAttachmentDescription
		{
			VkAttachmentDescription colorAttachment{};
			colorAttachment.format = getVKFormat(desc.format);
			colorAttachment.samples = getVkSampleCount(desc.samples);

			// determine what to do with the data in the attachment before rendering
			// - VK_ATTACHMENT_LOAD_OP_LOAD: Preserve the existing contents of the attachment
			// - VK_ATTACHMENT_LOAD_OP_CLEAR : Clear the values to a constant at the start
			// - VK_ATTACHMENT_LOAD_OP_DONT_CARE : Existing contents are undefined; we don't care about them
			colorAttachment.loadOp = getVkAttachmentLoadOp(desc.loadOp);

			// determine what to do with the data in the attachment after rendering
			// - VK_ATTACHMENT_STORE_OP_STORE: Rendered contents will be stored in memoryand can be read later
			// - VK_ATTACHMENT_STORE_OP_DONT_CARE : Contents of the framebuffer will be undefined after the rendering operation
			colorAttachment.storeOp = getVkAttachmentStorep(desc.storeOp);

			// apply to stencil data
			colorAttachment.stencilLoadOp = getVkAttachmentLoadOp(desc.stencilLoadOp);
			colorAttachment.stencilStoreOp = getVkAttachmentStorep(desc.stencilStoreOp);

			// the layout of the pixels in memory
			// - VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL: Images used as color attachment
			// - VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : Images to be presented in the swap chain
			// - VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : Images to be used as destination for a memory copy operation

			// The initialLayout specifies which layout the image will have before the render pass begins
			colorAttachment.initialLayout = getVkImageLayout(desc.initalLayout);
			// The finalLayout specifies the layout to automatically transition to when the render pass finishes
			colorAttachment.finalLayout = getVkImageLayout(desc.finalLayout);
			return colorAttachment;
		}

		auto createSubpass(
			std::vector<VkAttachmentReference> const& colorAttachmentRef, 
			std::vector<VkAttachmentReference> const& depthAttachmentRef, 
			VkSubpassDescription& subpass) noexcept -> void
		{
			// The subpass is described using a VkSubpassDescription structure :
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = colorAttachmentRef.size();
			subpass.pColorAttachments = colorAttachmentRef.data();
			if (depthAttachmentRef.size() != 0)
				subpass.pDepthStencilAttachment = depthAttachmentRef.data();

			// The index of the attachment in this array is directly referenced from the fragment shader with the layout(location = 0) out vec4 outColor directive!
			// - pInputAttachments: Attachments that are read from a shader
			// - pResolveAttachments: Attachments used for multisampling color attachments
			// - pDepthStencilAttachment : Attachment for depthand stencil data
			// - pPreserveAttachments : Attachments that are not used by this subpass, but for which the data must be preserved

		}

		auto createVKRenderPass(
			VkRenderPass& renderPass,
			VkAttachmentDescription* attachments,
			uint32_t const& attachment_num,
			VkSubpassDescription* subpasses,
			uint32_t const& subpass_num,
			bool has_depth,
			ILogicalDeviceVK* device
		) noexcept -> void
		{
			VkSubpassDependency dependency{};
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			if (has_depth)
			{
				dependency.srcStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
				dependency.dstStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
				dependency.dstAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			}

			VkRenderPassCreateInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.attachmentCount = attachment_num;
			renderPassInfo.pAttachments = attachments;
			renderPassInfo.subpassCount = subpass_num;
			renderPassInfo.pSubpasses = subpasses;
			renderPassInfo.dependencyCount = 1;
			renderPassInfo.pDependencies = &dependency;

			if (vkCreateRenderPass(device->getDeviceHandle(), &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
				SE_CORE_ERROR("VULKAN :: failed to create render pass!");
			}
		}

		IRenderPassVK::IRenderPassVK(RenderPassDesc const& desc, ILogicalDeviceVK* logical_device)
			:logicalDevice(logical_device)
		{
			createRenderPass(desc);
		}

		IRenderPassVK::~IRenderPassVK()
		{
			if (renderPass)
				vkDestroyRenderPass(logicalDevice->getDeviceHandle(), renderPass, nullptr);
		}

		auto IRenderPassVK::createRenderPass(RenderPassDesc const& desc) noexcept -> void
		{
			std::vector<VkAttachmentDescription> attachment_desc;
			attachment_desc.resize(desc.colorAttachments.size() + desc.depthstencialAttachments.size());
			clearValues.resize(attachment_desc.size());
			int idx = 0;
			for (auto const& attachment : desc.colorAttachments)
			{
				attachment_desc[idx] = createVkAttachmentDescriotion(desc.colorAttachments[idx]);
				ColorFloat4 const& color = desc.colorAttachments[idx].clearColor;
				clearValues[idx].color = { color.x,color.y,color.z,color.w };
				idx++;
			}
			int idx_depth = 0;
			for (auto const& attachment : desc.depthstencialAttachments)
			{
				ColorFloat4 const& color = desc.depthstencialAttachments[idx_depth].clearColor;
				clearValues[idx].depthStencil = { color.x,(uint32_t)color.y };
				attachment_desc[idx++] = createVkAttachmentDescriotion(desc.depthstencialAttachments[idx_depth++]);
			}

			int attachment = 0;
			std::vector<VkAttachmentReference> colorReferences(desc.colorAttachments.size());
			for (auto& color_reference : colorReferences)
			{
				color_reference.attachment = attachment++;
				color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
			}
			std::vector<VkAttachmentReference> depthReferences(desc.depthstencialAttachments.size());
			for (auto& ds_reference : depthReferences)
			{
				ds_reference.attachment = attachment++;
				ds_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
			}

			VkSubpassDescription subpass = {};
			createSubpass(colorReferences, depthReferences, subpass);

			createVKRenderPass(renderPass, attachment_desc.data(), attachment_desc.size(), &subpass, 1,
				depthReferences.size() != 0,
				logicalDevice);
		}

		auto IRenderPassVK::getRenderPass() noexcept -> VkRenderPass*
		{
			return &renderPass;
		}

		auto getVkAttachmentLoadOp(AttachmentLoadOp op) noexcept -> VkAttachmentLoadOp
		{
			switch (op)
			{
			case SIByL::RHI::AttachmentLoadOp::DONT_CARE:
				return VK_ATTACHMENT_LOAD_OP_DONT_CARE;
				break;
			case SIByL::RHI::AttachmentLoadOp::CLEAR:
				return VK_ATTACHMENT_LOAD_OP_CLEAR;
				break;
			case SIByL::RHI::AttachmentLoadOp::LOAD:
				return VK_ATTACHMENT_LOAD_OP_LOAD;
				break;
			default:
				break;
			}
			return VK_ATTACHMENT_LOAD_OP_MAX_ENUM;
		}

		auto getVkAttachmentStorep(AttachmentStoreOp op) noexcept -> VkAttachmentStoreOp
		{
			switch (op)
			{
			case SIByL::RHI::AttachmentStoreOp::DONT_CARE:
				return VK_ATTACHMENT_STORE_OP_DONT_CARE;
				break;
			case SIByL::RHI::AttachmentStoreOp::STORE:
				return VK_ATTACHMENT_STORE_OP_STORE;
				break;
			default:
				break;
			}
			return VK_ATTACHMENT_STORE_OP_MAX_ENUM;
		}
	}
}
