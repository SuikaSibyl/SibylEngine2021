module;
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IFramebuffer.VK;
import Core.Log;
import Core.Color;
import RHI.IResource;
import RHI.IFramebuffer;
import RHI.IRenderPass;
import RHI.IRenderPass.VK;
import RHI.ITexture;
import RHI.ITexture.VK;
import RHI.ITextureView;
import RHI.ITextureView.VK;

namespace SIByL::RHI
{
    IFramebufferVK::IFramebufferVK(FramebufferDesc const& _desc, ILogicalDeviceVK* _logical_device)
        : desc(_desc)
        , logicalDevice(_logical_device)
    {
        createFramebuffers();
    }

    IFramebufferVK::~IFramebufferVK()
    {
        if(framebuffer)
            vkDestroyFramebuffer(logicalDevice->getDeviceHandle(), framebuffer, nullptr);
    }

    auto IFramebufferVK::getVkFramebuffer() noexcept -> VkFramebuffer*
    {
        return &framebuffer;
    }

    auto IFramebufferVK::getSize(uint32_t& _width, uint32_t& _height) noexcept -> void
    {
        _width = desc.width;
        _height = desc.height;
    }

	auto IFramebufferVK::createFramebuffers() noexcept -> void
	{
        std::vector<VkImageView> attachments;
        for (int i = 0; i < desc.attachments.size(); i++)
        {
            attachments.emplace_back(*((ITextureViewVK*)desc.attachments[i])->getpVkImageView());
        }

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = *((IRenderPassVK*)desc.renderPass)->getRenderPass();
        framebufferInfo.attachmentCount = attachments.size();
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = desc.width;
        framebufferInfo.height = desc.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(logicalDevice->getDeviceHandle(), &framebufferInfo, nullptr, 
            &framebuffer) != VK_SUCCESS) {
            SE_CORE_ERROR("failed to create framebuffer!");
        }
	}
}
