module;
#include <cstdint>
#include <vulkan/vulkan.h>
export module RHI.IMemoryBarrier.VK;
import RHI.IMemoryBarrier;
import RHI.IEnum;
import RHI.ICommandQueue;
import RHI.ITexture;
import RHI.IBuffer;

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

        export class IBufferMemoryBarrierVK :public IBufferMemoryBarrier
        {
        public:
            IBufferMemoryBarrierVK(BufferMemoryBarrierDesc const&);
            virtual ~IBufferMemoryBarrierVK() = default;
            auto getVkBufferMemoryBarrier() noexcept -> VkBufferMemoryBarrier*;

        private:
            VkBufferMemoryBarrier barrier{};
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