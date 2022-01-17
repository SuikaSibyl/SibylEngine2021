module;
#include <optional>
#include <vulkan/vulkan.h>
module RHI.LogicalDevice.VK;
import RHI.GraphicContext;
import RHI.GraphicContext.VK;

namespace SIByL::RHI
{
	auto ILogicalDeviceVK::createLogicalDevice(IGraphicContextVK& context) -> void
	{
		QueueFamilyIndicesVK indices = context.findQueueFamilies();

		VkDeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
		queueCreateInfo.queueCount = 1;

		float queuePriority = 1.0f;
		queueCreateInfo.pQueuePriorities = &queuePriority;

		VkPhysicalDeviceFeatures deviceFeatures{};

		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = &queueCreateInfo;
		createInfo.queueCreateInfoCount = 1;
		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = 0;

		//if (context.hasValidationLayers()) {
		//	createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
		//	createInfo.ppEnabledLayerNames = validationLayers.data();
		//}
		//else {
		//	createInfo.enabledLayerCount = 0;
		//}

	}

}
