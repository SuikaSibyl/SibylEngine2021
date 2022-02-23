module;
#include <vulkan/vulkan.h>
export module RHI.IBuffer.VK;
import RHI.IBuffer;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IBufferVK :public IBuffer
		{
		public:
			IBufferVK() = default;
			IBufferVK(BufferDesc const& desc, ILogicalDeviceVK* logical_device);
			IBufferVK(IBufferVK&&);
			IBufferVK(IBufferVK const&) = delete;
			virtual ~IBufferVK();

			auto operator = (IBufferVK const& desc) -> IBufferVK& = delete;
			auto operator = (IBufferVK && desc) -> IBufferVK&;

			virtual auto getSize() noexcept -> uint32_t override;

			auto getVkBuffer() noexcept -> VkBuffer*;
			auto getVkDeviceMemory() noexcept -> VkDeviceMemory*;
			
		private:
			auto release() noexcept -> void;
			uint32_t size;
			VkBuffer buffer = {};
			VkDeviceMemory bufferMemory = {};
			ILogicalDeviceVK* logicalDevice;
		};

	}
}