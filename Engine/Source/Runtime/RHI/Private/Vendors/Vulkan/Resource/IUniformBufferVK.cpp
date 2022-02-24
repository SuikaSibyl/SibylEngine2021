module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IUniformBuffer.VK;
import Core.Buffer;
import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;
import RHI.IBuffer.VK;
import RHI.IUniformBuffer;
import RHI.ILogicalDevice.VK;

namespace SIByL::RHI
{
	IUniformBufferVK::IUniformBufferVK(uint32_t const& size, ILogicalDeviceVK* logical_device)
		: logicalDevice(logical_device)
	{
		// create vertex buffer
		BufferDesc bufferDesc =
		{
			(unsigned int)size,
			(BufferUsageFlags)BufferUsageFlagBits::UNIFORM_BUFFER_BIT,
			BufferShareMode::EXCLUSIVE,
			(uint32_t)MemoryPropertyFlagBits::HOST_VISIBLE_BIT | (uint32_t)MemoryPropertyFlagBits::HOST_COHERENT_BIT
		};
		buffer = std::move(IBufferVK(bufferDesc, logicalDevice));
	}

	auto IUniformBufferVK::updateBuffer(Buffer* _buffer) noexcept -> void
	{
		void* data;
		vkMapMemory(logicalDevice->getDeviceHandle(), *(buffer.getVkDeviceMemory()), 0, buffer.getSize(), 0, &data);
		memcpy(data, _buffer->getData(), (unsigned int)_buffer->getSize());
		vkUnmapMemory(logicalDevice->getDeviceHandle(), *(buffer.getVkDeviceMemory()));
	}

	auto IUniformBufferVK::getSize() noexcept -> uint32_t
	{
		return buffer.getSize();
	}

	auto IUniformBufferVK::getVkBuffer() noexcept -> VkBuffer*
	{
		return buffer.getVkBuffer();
	}
}