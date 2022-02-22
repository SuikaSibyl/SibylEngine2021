module;
#include <vulkan/vulkan.h>
#include <vector>
#include <set>
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
	
	auto ILogicalDeviceVK::getPhysicalDevice() noexcept -> IPhysicalDevice*
	{
		return (IPhysicalDevice*)physicalDevice;
	}

	auto ILogicalDeviceVK::waitIdle() noexcept -> void
	{
		vkDeviceWaitIdle(device);
	}

	auto ILogicalDeviceVK::getDeviceHandle() noexcept -> VkDevice&
	{
		return device;
	}

	auto ILogicalDeviceVK::getPhysicalDeviceVk() noexcept -> IPhysicalDeviceVK*
	{
		return physicalDevice;
	}

	auto ILogicalDeviceVK::getVkGraphicQueue() noexcept -> VkQueue*
	{
		return &graphicsQueue;
	}
	
	auto ILogicalDeviceVK::getVkPresentQueue() noexcept -> VkQueue*
	{
		return &presentQueue;
	}

	auto ILogicalDeviceVK::createLogicalDevice(IPhysicalDeviceVK* physicalDevice) noexcept -> void
	{
		IPhysicalDeviceVK::QueueFamilyIndices indices = physicalDevice -> findQueueFamilies();

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		// Desc VkDeviceQueueCreateInfo
		VkDeviceQueueCreateInfo queueCreateInfo{};
		// the number of queues we want for a single queue family
		float queuePriority = 1.0f;	// a queue with graphics capabilities
		for (uint32_t queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			queueCreateInfos.push_back(queueCreateInfo);
		}

		// Desc Vk Physical Device Features
		// - the set of device features that we'll be using
		VkPhysicalDeviceFeatures deviceFeatures{};

		// Desc Vk Device Create Info
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pEnabledFeatures = &deviceFeatures;
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		createInfo.pQueueCreateInfos = queueCreateInfos.data();
		// enable extesions
		createInfo.enabledExtensionCount = static_cast<uint32_t>(physicalDevice->getDeviceExtensions().size());
		createInfo.ppEnabledExtensionNames = physicalDevice->getDeviceExtensions().data();

		if (vkCreateDevice(physicalDevice->getPhysicalDevice(), &createInfo, nullptr, &device) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to create logical device!");
		}

		// get queue handle
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

}
