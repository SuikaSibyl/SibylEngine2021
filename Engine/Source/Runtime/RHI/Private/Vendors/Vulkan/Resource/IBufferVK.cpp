module;
#include <vulkan/vulkan.h>
module RHI.IBuffer.VK;
import Core.Log;
import RHI.IBuffer;
import RHI.ILogicalDevice.VK;
import RHI.IPhysicalDevice.VK;
import RHI.IEnum;
import RHI.IEnum.VK;

namespace SIByL::RHI
{
    void createBuffer(
        VkDeviceSize size, 
        VkBufferUsageFlags usage, 
        VkSharingMode shareMode,
        VkMemoryPropertyFlags properties, 
        VkBuffer& buffer, 
        VkDeviceMemory& bufferMemory,
        ILogicalDeviceVK* device) 
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = shareMode;

        if (vkCreateBuffer(device->getDeviceHandle(), &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            SE_CORE_ERROR("VULKAN :: failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device->getDeviceHandle(), buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = device->getPhysicalDeviceVk()->findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(device->getDeviceHandle(), &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            SE_CORE_ERROR("VULKAN :: failed to allocate buffer memory!");
        }

        vkBindBufferMemory(device->getDeviceHandle(), buffer, bufferMemory, 0);
    }

    IBufferVK::IBufferVK(BufferDesc const& desc, ILogicalDeviceVK* logical_device)
        : logicalDevice(logical_device)
        , size(desc.size)
    {
        createBuffer(
            desc.size, 
            getVkBufferUsage(desc.usage), 
            getVkBufferShareMode(desc.shareMode),
            getVkMemoryProperty(desc.memoryProperty),
            buffer,
            bufferMemory,
            logicalDevice
        );
    }

    IBufferVK::~IBufferVK()
    {
        release();
    }

    IBufferVK::IBufferVK(IBufferVK&& rho)
    {
        release();
        buffer = rho.buffer;
        bufferMemory = rho.bufferMemory;
        logicalDevice = rho.logicalDevice;
        size = rho.size;

        rho.buffer = nullptr;
        rho.bufferMemory = nullptr;
        rho.logicalDevice = nullptr;
        rho.size = 0;
    }

    auto IBufferVK::operator= (IBufferVK&& rho)->IBufferVK&
    {
        release();
        buffer = rho.buffer;
        bufferMemory = rho.bufferMemory;
        logicalDevice = rho.logicalDevice;
        size = rho.size;

        rho.buffer = nullptr;
        rho.bufferMemory = nullptr;
        rho.logicalDevice = nullptr;
        rho.size = 0;

        return *this;
    }
    
    auto IBufferVK::getSize() noexcept -> uint32_t
    {
        return size;
    }

    auto IBufferVK::getVkBuffer() noexcept -> VkBuffer*
    {
        return &buffer;
    }

    auto IBufferVK::getVkDeviceMemory() noexcept -> VkDeviceMemory*
    {
        return &bufferMemory;
    }

    auto IBufferVK::release() noexcept -> void
    {
        if (buffer)
            vkDestroyBuffer(logicalDevice->getDeviceHandle(), buffer, nullptr);
        if (bufferMemory)
            vkFreeMemory(logicalDevice->getDeviceHandle(), bufferMemory, nullptr);
    }
}