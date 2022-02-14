module;
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IFramebuffer.VK;
import Core.Log;
import RHI.IResource;
import RHI.IFramebuffer;
import RHI.IRenderPass;
import RHI.IRenderPass.VK;

namespace SIByL::RHI
{
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
        _width = width;
        _height = height;
    }

	auto IFramebufferVK::createFramebuffers() noexcept -> void
	{
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = *renderPass->getRenderPass();
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = width;
        framebufferInfo.height = height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(logicalDevice->getDeviceHandle(), &framebufferInfo, nullptr, 
            &framebuffer) != VK_SUCCESS) {
            SE_CORE_ERROR("failed to create framebuffer!");
        }
	}
}
