module;
#include <cstdint>
#include <vulkan/vulkan.h>
module RHI.IMemoryBarrier.VK;
import RHI.IMemoryBarrier;
import RHI.IEnum;
import RHI.IEnum.VK;
import RHI.ICommandQueue;
import RHI.ITexture;
import RHI.ICommandQueue;
import RHI.ITexture;
import RHI.ITexture.VK;

//import RHI.ICommandQueue.VK;

namespace SIByL::RHI
{
    IMemoryBarrierVK::IMemoryBarrierVK(MemoryBarrierDesc const& desc)
    {
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = getVkAccessFlags(desc.srcAccessMask);
        barrier.dstAccessMask = getVkAccessFlags(desc.dstAccessMask);
    }
    
    auto IMemoryBarrierVK::getVkMemoryBarrier() noexcept -> VkMemoryBarrier*
    {
        return &barrier;
    }

    IImageMemoryBarrierVK::IImageMemoryBarrierVK(ImageMemoryBarrierDesc const& desc)
    {
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = getVkImageLayout(desc.oldLayout);
        barrier.newLayout = getVkImageLayout(desc.newLayout);
        barrier.srcQueueFamilyIndex = desc.srcQueue == nullptr ? VK_QUEUE_FAMILY_IGNORED : VK_QUEUE_FAMILY_IGNORED; // TODO: FIX QUEUE TRANSITION
        barrier.dstQueueFamilyIndex = desc.dstQueue == nullptr ? VK_QUEUE_FAMILY_IGNORED : VK_QUEUE_FAMILY_IGNORED; // TODO: FIX QUEUE TRANSITION
        barrier.image = *((ITextureVK*)desc.image)->getVkImage();
        barrier.subresourceRange.aspectMask = getVkImageAspectFlags(desc.subresourceRange.aspectMask);
        barrier.subresourceRange.baseMipLevel = desc.subresourceRange.baseMipLevel;
        barrier.subresourceRange.levelCount = desc.subresourceRange.levelCount;
        barrier.subresourceRange.baseArrayLayer = desc.subresourceRange.baseArrayLayer;
        barrier.subresourceRange.layerCount = desc.subresourceRange.levelCount;
        barrier.srcAccessMask = getVkAccessFlags(desc.srcAccessMask);
        barrier.dstAccessMask = getVkAccessFlags(desc.dstAccessMask);
    }

    auto IImageMemoryBarrierVK::getVkImageMemoryBarrier() noexcept -> VkImageMemoryBarrier*
    {
        return &barrier;
    }

    auto getVkAccessFlags(AccessFlags _flag) noexcept -> VkAccessFlags
    {
        uint32_t flags{};
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::INDIRECT_COMMAND_READ_BIT,                 VK_ACCESS_INDIRECT_COMMAND_READ_BIT,                     flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::INDEX_READ_BIT,                            VK_ACCESS_INDEX_READ_BIT,                                flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::VERTEX_ATTRIBUTE_READ_BIT,                 VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,                     flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::UNIFORM_READ_BIT,                          VK_ACCESS_UNIFORM_READ_BIT,                              flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::INPUT_ATTACHMENT_READ_BIT,                 VK_ACCESS_INPUT_ATTACHMENT_READ_BIT,                     flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::SHADER_READ_BIT,                           VK_ACCESS_SHADER_READ_BIT,                               flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::SHADER_WRITE_BIT,                          VK_ACCESS_SHADER_WRITE_BIT,                              flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::COLOR_ATTACHMENT_READ_BIT,                 VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,                     flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT,                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,                    flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT,         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,             flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,            flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::TRANSFER_READ_BIT,                         VK_ACCESS_TRANSFER_READ_BIT,                             flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::TRANSFER_WRITE_BIT,                        VK_ACCESS_TRANSFER_WRITE_BIT,                            flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::HOST_READ_BIT,                             VK_ACCESS_HOST_READ_BIT,                                 flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::HOST_WRITE_BIT,                            VK_ACCESS_HOST_WRITE_BIT,                                flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::MEMORY_READ_BIT,                           VK_ACCESS_MEMORY_READ_BIT,                               flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::MEMORY_WRITE_BIT,                          VK_ACCESS_MEMORY_WRITE_BIT,                              flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::TRANSFORM_FEEDBACK_WRITE_BIT,              VK_ACCESS_TRANSFORM_FEEDBACK_WRITE_BIT_EXT,              flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::TRANSFORM_FEEDBACK_COUNTER_READ_BIT,       VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT,       flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT,      VK_ACCESS_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,      flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::CONDITIONAL_RENDERING_READ_BIT,            VK_ACCESS_CONDITIONAL_RENDERING_READ_BIT_EXT,            flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::COLOR_ATTACHMENT_READ_NONCOHERENT_BIT,     VK_ACCESS_COLOR_ATTACHMENT_READ_NONCOHERENT_BIT_EXT,     flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::ACCELERATION_STRUCTURE_READ_BIT,           VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,           flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::ACCELERATION_STRUCTURE_WRITE_BIT,          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,          flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::FRAGMENT_DENSITY_MAP_READ_BIT,             VK_ACCESS_FRAGMENT_DENSITY_MAP_READ_BIT_EXT,             flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT, VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR, flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::COMMAND_PREPROCESS_READ_BIT,               VK_ACCESS_COMMAND_PREPROCESS_READ_BIT_NV,                flags);
        flagBitSwitch(_flag, (uint32_t)SIByL::RHI::AccessFlagBits::COMMAND_PREPROCESS_WRITE_BIT,              VK_ACCESS_COMMAND_PREPROCESS_WRITE_BIT_NV,               flags);
        return (VkShaderStageFlags)flags;
    }

}