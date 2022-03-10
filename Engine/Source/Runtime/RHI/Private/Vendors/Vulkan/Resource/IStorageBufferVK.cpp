module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IStorageBuffer.VK;
import Core.Buffer;
import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;
import RHI.IBuffer.VK;
import RHI.IStorageBuffer;
import RHI.ILogicalDevice.VK;

namespace SIByL::RHI
{
	IStorageBufferVK::IStorageBufferVK(uint32_t const& size, ILogicalDeviceVK* logical_device, BufferUsageFlags const& extra_usage)
		: logicalDevice(logical_device)
	{
		BufferDesc bufferDesc =
		{
			(unsigned int)size,
			(BufferUsageFlags)BufferUsageFlagBits::STORAGE_BUFFER_BIT | extra_usage,
			BufferShareMode::EXCLUSIVE,
			(uint32_t)MemoryPropertyFlagBits::DEVICE_LOCAL_BIT
		};
		buffer = std::move(IBufferVK(bufferDesc, logicalDevice));
	}

	auto IStorageBufferVK::getSize() noexcept -> uint32_t
	{
		return buffer.getSize();
	}

	auto IStorageBufferVK::getVkBuffer() noexcept -> VkBuffer*
	{
		return buffer.getVkBuffer();
	}
}