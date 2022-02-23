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
import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;
import RHI.IVertexBuffer;
import RHI.IEnum.VK;
import RHI.ILogicalDevice.VK;

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

		IVertexBufferVK::IVertexBufferVK(Buffer* buffer, ILogicalDeviceVK* logical_device)
			: logicalDevice(logical_device)
		{
			createVertexBuffer(logicalDevice, buffer, &vertexBuffer);

			// alloc memory
			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(logicalDevice->getDeviceHandle(), vertexBuffer, &memRequirements);
			logicalDevice->allocMemory(&memRequirements, &vertexBuffer, &vertexBufferMemory);

			// copy memory
			void* data;
			vkMapMemory(logicalDevice->getDeviceHandle(), vertexBufferMemory, 0, buffer->getSize(), 0, &data);
			memcpy(data, buffer->getData(), (size_t)buffer->getSize());
			vkUnmapMemory(logicalDevice->getDeviceHandle(), vertexBufferMemory);

		}

		auto IVertexBufferVK::getVkBuffer() noexcept ->VkBuffer*
		{
			return &vertexBuffer;
		}

		IVertexBufferVK::~IVertexBufferVK()
		{
			if(vertexBuffer)
				vkDestroyBuffer(logicalDevice->getDeviceHandle(), vertexBuffer, nullptr);
			if (vertexBufferMemory)
				vkFreeMemory(logicalDevice->getDeviceHandle(), vertexBufferMemory, nullptr);
		}
	}
}