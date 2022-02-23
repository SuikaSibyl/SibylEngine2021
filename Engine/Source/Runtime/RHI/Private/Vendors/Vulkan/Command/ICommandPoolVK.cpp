module;
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
module RHI.ICommandPool.VK;
import Core.Log;
import RHI.ICommandPool;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
import RHI.IPhysicalDevice;
import RHI.IPhysicalDevice.VK;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		ICommandPoolVK::ICommandPoolVK(CommandPoolDesc const& desc, ILogicalDeviceVK* logical_device)
			: logicalDevice(logical_device)
			, type(desc.type)
		{
			createVkCommandPool();
		}

		ICommandPoolVK::~ICommandPoolVK()
		{
			if(commandPool)
				vkDestroyCommandPool(logicalDevice->getDeviceHandle(), commandPool, nullptr);
		}

		auto ICommandPoolVK::getVkCommandPool() noexcept -> VkCommandPool*
		{
			return &commandPool;
		}

		auto ICommandPoolVK::createVkCommandPool() noexcept -> void
		{
			IPhysicalDeviceVK* physical_device = logicalDevice->getPhysicalDeviceVk();
			IPhysicalDeviceVK::QueueFamilyIndices queueFamilyIndices = physical_device->findQueueFamilies();

			VkCommandPoolCreateInfo poolInfo{};
			poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			switch (type)
			{
			case SIByL::RHI::QueueType::PRESENTATION:
				break;
			case SIByL::RHI::QueueType::GRAPHICS:
				poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
				break;
			default:
				break;
			}

			// VK_COMMAND_POOL_CREATE_TRANSIENT_BIT: 
			// - Hint that command buffers are rerecorded with new commands very often(may change memory allocation behavior)
			// VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT : 
			// - Allow command buffers to be rerecorded individually, without this flag they all have to be reset together
			poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

			if (vkCreateCommandPool(logicalDevice->getDeviceHandle(), &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
				SE_CORE_ERROR("VULKAN :: failed to create command pool!");
			}
		}
	}
}
