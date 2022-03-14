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
import RHI.IDeviceGlobal;
import RHI.ICommandBuffer;
import RHI.ICommandPool;

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

	IStorageBufferVK::IStorageBufferVK(Buffer* _buffer, ILogicalDeviceVK* logical_device, BufferUsageFlags const& extra_usage)
		: logicalDevice(logical_device)
	{
		// create staging buffer
		BufferDesc stagingBufferDesc =
		{
			(unsigned int)_buffer->getSize(),
			(BufferUsageFlags)BufferUsageFlagBits::TRANSFER_SRC_BIT,
			BufferShareMode::EXCLUSIVE,
			(uint32_t)MemoryPropertyFlagBits::HOST_VISIBLE_BIT | (uint32_t)MemoryPropertyFlagBits::HOST_COHERENT_BIT
		};
		IBufferVK stagingBuffer(stagingBufferDesc, logicalDevice);

		// copy memory
		void* data;
		vkMapMemory(logicalDevice->getDeviceHandle(), *(stagingBuffer.getVkDeviceMemory()), 0, stagingBuffer.getSize(), 0, &data);
		memcpy(data, _buffer->getData(), (unsigned int)_buffer->getSize());
		vkUnmapMemory(logicalDevice->getDeviceHandle(), *(stagingBuffer.getVkDeviceMemory()));

		// create vertex buffer
		BufferDesc bufferDesc =
		{
			(unsigned int)_buffer->getSize(),
			(BufferUsageFlags)BufferUsageFlagBits::STORAGE_BUFFER_BIT | (BufferUsageFlags)BufferUsageFlagBits::TRANSFER_DST_BIT | extra_usage,
			BufferShareMode::EXCLUSIVE,
			(uint32_t)MemoryPropertyFlagBits::DEVICE_LOCAL_BIT
		};
		buffer = std::move(IBufferVK(bufferDesc, logicalDevice));

		// copy
		PerDeviceGlobal* global = DeviceToGlobal::getGlobal((ILogicalDevice*)logicalDevice);
		ICommandPool* transientPool = global->getTransientCommandPool();
		MemScope<ICommandBuffer> commandbuffer = global->getResourceFactory()->createCommandBuffer(transientPool);
		commandbuffer->beginRecording((uint32_t)CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT);
		commandbuffer->cmdCopyBuffer((IBuffer*)&stagingBuffer, (IBuffer*)&buffer, 0, 0, buffer.getSize());
		commandbuffer->endRecording();
		commandbuffer->submit();
		logicalDevice->waitIdle();
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