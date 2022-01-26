module;
#include <vulkan/vulkan.h>
#include <vector>
module RHI.ILogicalDevice.VK;
import Core.Log;
import RHI.IPhysicalDevice.VK;

namespace SIByL::RHI
{
	ILogicalDeviceVK::ILogicalDeviceVK(IPhysicalDeviceVK* physicalDevice)
		:physicalDevice(physicalDevice)
	{

	}

	auto ILogicalDeviceVK::initialize() -> bool
	{
		createLogicalDevice(physicalDevice);
		return true;
	}

	auto ILogicalDeviceVK::destroy() -> bool
	{
		vkDestroyDevice(device, nullptr);
		return true;
	}

	auto ILogicalDeviceVK::createLogicalDevice(IPhysicalDeviceVK* physicalDevice) noexcept -> void
	{
		IPhysicalDeviceVK::QueueFamilyIndices indices = physicalDevice -> findQueueFamilies();

		// Desc VkDeviceQueueCreateInfo
		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
		// the number of queues we want for a single queue family
		queueCreateInfo.queueCount = 1; // a queue with graphics capabilities
		// influence the scheduling of command buffer execution
		float queuePriority = 1.0f;
		queueCreateInfo.pQueuePriorities = &queuePriority;

		// Desc Vk Physical Device Features
		// - the set of device features that we'll be using
		VkPhysicalDeviceFeatures deviceFeatures{};

		// Desc Vk Device Create Info
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = &queueCreateInfo;
		createInfo.queueCreateInfoCount = 1;
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.enabledExtensionCount = 0;

		if (vkCreateDevice(physicalDevice->getPhysicalDevice(), &createInfo, nullptr, &device) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to create logical device!");
		}

		// get queue handle
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
	}

}
