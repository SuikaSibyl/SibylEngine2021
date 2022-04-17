module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IIndexBuffer.VK;

import Core.Buffer;
import Core.Log;
import Core.MemoryManager;

import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;
import RHI.IEnum.VK;
import RHI.ILogicalDevice.VK;
import RHI.IBuffer;
import RHI.IBuffer.VK;
import RHI.IDeviceGlobal;
import RHI.ICommandPool;
import RHI.ICommandBuffer;
import RHI.IFactory;
import RHI.IIndexBuffer;

namespace SIByL::RHI
{
	IIndexBufferVK::IIndexBufferVK(Buffer* _buffer, uint32_t const& size, ILogicalDeviceVK* logical_device)
		: logicalDevice(logical_device)
		, elementSize(size)
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
		indicesCount = (uint32_t)_buffer->getSize() / elementSize;

		// create vertex buffer
		BufferDesc bufferDesc =
		{
			(unsigned int)_buffer->getSize(),
			(BufferUsageFlags)BufferUsageFlagBits::INDEX_BUFFER_BIT | (BufferUsageFlags)BufferUsageFlagBits::TRANSFER_DST_BIT,
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
	
	auto IIndexBufferVK::getElementSize() noexcept -> uint32_t
	{
		return elementSize;
	}
	
	auto IIndexBufferVK::getIndicesCount() noexcept -> uint32_t
	{
		return indicesCount;
	}

	auto IIndexBufferVK::getVkBuffer() noexcept ->VkBuffer*
	{
		return buffer.getVkBuffer();
	}

	auto IIndexBufferVK::getVkIndexType() noexcept -> VkIndexType
	{
		if (elementSize == 2)
		{
			return VK_INDEX_TYPE_UINT16;
		}
		else if (elementSize == 4)
		{
			return VK_INDEX_TYPE_UINT32;
		}
		else
		{
			SE_CORE_ERROR("VULKAN :: unsupported index size");
		}
	}
}