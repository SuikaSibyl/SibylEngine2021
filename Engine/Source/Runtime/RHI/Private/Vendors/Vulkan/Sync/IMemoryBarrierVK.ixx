module;
#include <cstdint>
#include <vulkan/vulkan.h>
export module RHI.IMemoryBarrier.VK;
import RHI.IMemoryBarrier;
import RHI.IEnum;
import RHI.ICommandQueue;
import RHI.ITexture;

namespace SIByL
{
	namespace RHI
	{
        export class IMemoryBarrierVK :public IMemoryBarrier
        {
        public:
            IMemoryBarrierVK(MemoryBarrierDesc const&);
            virtual ~IMemoryBarrierVK() = default;
            auto getVkMemoryBarrier() noexcept -> VkMemoryBarrier*;

        private:
            VkMemoryBarrier barrier{};
        };

        export class IImageMemoryBarrierVK: public IImageMemoryBarrier
        {
        public:
            IImageMemoryBarrierVK(ImageMemoryBarrierDesc const&);
            virtual ~IImageMemoryBarrierVK() = default;
            auto getVkImageMemoryBarrier() noexcept -> VkImageMemoryBarrier*;

        private:
            VkImageMemoryBarrier barrier{};
        };

        export auto getVkAccessFlags(AccessFlags) noexcept -> VkAccessFlags;
	}
}