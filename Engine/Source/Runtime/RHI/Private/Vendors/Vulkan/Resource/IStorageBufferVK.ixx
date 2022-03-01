module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IStorageBuffer.VK;
import Core.Buffer;
import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;
import RHI.IBuffer.VK;
import RHI.IStorageBuffer;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IStorageBufferVK :public IStorageBuffer
		{
		public:
			IStorageBufferVK(uint32_t const& size, ILogicalDeviceVK* logical_device);
			virtual ~IStorageBufferVK() = default;

			virtual auto getSize() noexcept -> uint32_t override;
			auto getVkBuffer() noexcept -> VkBuffer*;

		private:
			IBufferVK buffer;
			ILogicalDeviceVK* logicalDevice;
		};
	}
}