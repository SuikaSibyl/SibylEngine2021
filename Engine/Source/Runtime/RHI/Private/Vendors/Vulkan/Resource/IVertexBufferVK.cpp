module;
#include <cstdint>
#include <string>
#include <string_view>
#include <array>
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IVertexBuffer.VK;
import Core.Buffer;
import Core.Log;
import Core.MemoryManager;

import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;
import RHI.IVertexBuffer;
import RHI.IEnum.VK;
import RHI.ILogicalDevice.VK;
import RHI.IBuffer;
import RHI.IBuffer.VK;
import RHI.IDeviceGlobal;
import RHI.ICommandPool;
import RHI.ICommandBuffer;
import RHI.IFactory;

namespace SIByL
{
	namespace RHI
	{
		auto getVertexBufferCreateInfo(Buffer* buffer) noexcept -> VkBufferCreateInfo
		{
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = buffer->getSize();
			bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
			return bufferInfo;
		}

		auto createVertexBuffer(ILogicalDeviceVK* logical_device, Buffer* buffer, VkBuffer* vertexBuffer) noexcept -> void
		{
			VkBufferCreateInfo bufferInfo = getVertexBufferCreateInfo(buffer);
			if (vkCreateBuffer(logical_device->getDeviceHandle(), &bufferInfo, nullptr, vertexBuffer) != VK_SUCCESS) {
				SE_CORE_ERROR("VULKAN :: failed to create vertex buffer!");
			}
		}

		IVertexBufferVK::IVertexBufferVK(Buffer* _buffer, ILogicalDeviceVK* logical_device)
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
				(BufferUsageFlags)BufferUsageFlagBits::VERTEX_BUFFER_BIT | (BufferUsageFlags)BufferUsageFlagBits::TRANSFER_DST_BIT,
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

		auto IVertexBufferVK::getVkBuffer() noexcept ->VkBuffer*
		{
			return buffer.getVkBuffer();
		}

		IVertexBufferVK::~IVertexBufferVK()
		{

		}
	}
}