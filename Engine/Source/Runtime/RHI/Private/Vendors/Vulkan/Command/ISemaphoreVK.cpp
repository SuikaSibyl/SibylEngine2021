module;
#include <vulkan/vulkan.h>
module RHI.ISemaphore.VK;
import Core.Log;
import RHI.ISemaphore;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		ISemaphoreVK::ISemaphoreVK(ILogicalDeviceVK* logical_device)
			:logicalDevice(logical_device)
		{
			createSemaphores();
		}

		ISemaphoreVK::~ISemaphoreVK()
		{
			if (semaphore)
			{
				vkDestroySemaphore(logicalDevice->getDeviceHandle(), semaphore, nullptr);
			}
		}

		auto ISemaphoreVK::getVkSemaphore() noexcept -> VkSemaphore*
		{
			return &semaphore;
		}

		auto ISemaphoreVK::createSemaphores() noexcept -> void
		{
			VkSemaphoreCreateInfo semaphoreInfo{};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
			if (vkCreateSemaphore(logicalDevice->getDeviceHandle(), &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS)
			{
				SE_CORE_ERROR("VULKAN :: failed to create semaphores!");
			}
		}
	}
}
