module;
#include <cstdint>
#include <vulkan/vulkan.h>
module RHI.IMemoryBarrierVK;
import RHI.IMemoryBarrier;
import RHI.IEnum;
import RHI.ICommandQueue;
import RHI.ITexture;

namespace SIByL::RHI
{
    IImageMemoryBarrierVK::IImageMemoryBarrierVK(ImageMemoryBarrierDesc const&)
    {

    }

    auto IImageMemoryBarrierVK::getVkImageMemoryBarrier() noexcept -> VkImageMemoryBarrier*
    {
        return &barrier;
    }
}