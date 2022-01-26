module;
#include <vulkan/vulkan.h>
#include <vector>
module RHI.IPhysicalDevice.VK;
import Core.Log;
import RHI.GraphicContext.VK;

namespace SIByL::RHI
{
#ifdef _DEBUG
	const bool enableValidationLayers = true;
#else
	const bool enableValidationLayers = false;
#endif
	
	auto IPhysicalDeviceVK::initialize() -> bool
	{
		enableDebugLayer = enableValidationLayers;
		queryAllPhysicalDevice();
		return true;
	}

	auto IPhysicalDeviceVK::isDeviceSuitable(VkPhysicalDevice device) -> bool
	{
		VkPhysicalDeviceProperties deviceProperties;
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		return true;
		//QueueFamilyIndicesVK indices = findQueueFamilies(device);
		//return indices.isComplete();
	}

	auto IPhysicalDeviceVK::rateDeviceSuitability(VkPhysicalDevice device) -> int
	{
		VkPhysicalDeviceProperties deviceProperties;
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

		int score = 0;

		// Discrete GPUs have a significant performance advantage
		if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			score += 1000;
		}

		// Maximum possible size of textures affects graphics quality
		score += deviceProperties.limits.maxImageDimension2D;

		// Application can't function without geometry shaders
		if (!deviceFeatures.geometryShader) {
			return 0;
		}

		return score;
	}

	auto IPhysicalDeviceVK::isDebugLayerEnabled() noexcept -> bool
	{
		return enableDebugLayer;
	}

	auto IPhysicalDeviceVK::queryAllPhysicalDevice() noexcept -> void
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(IGraphicContextVK::getVKInstance(), &deviceCount, nullptr);

		// If there are 0 devices with Vulkan support
		if (deviceCount == 0) {
			SE_CORE_ERROR("VULKAN :: failed to find GPUs with Vulkan support!");
		}

		// get all of the VkPhysicalDevice handles
		devices.resize(deviceCount);
		vkEnumeratePhysicalDevices(IGraphicContextVK::getVKInstance(), &deviceCount, devices.data());

		// check if any of the physical devices meet the requirements
		for (const auto& device : devices) {
			VkPhysicalDeviceProperties deviceProperties;
			vkGetPhysicalDeviceProperties(device, &deviceProperties);

			SE_CORE_INFO("VULKAN :: Physical Device Found, {0}", deviceProperties.deviceName);
		}

		// Find the best
		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE) {
			SE_CORE_ERROR("VULKAN :: failed to find a suitable GPU!");
		}
	}

	auto IPhysicalDeviceVK::getPhysicalDevice() noexcept -> VkPhysicalDevice&
	{
		return physicalDevice;
	}

	auto IPhysicalDeviceVK::findQueueFamilies() -> QueueFamilyIndices
	{
		QueueFamilyIndices indices;
		// Logic to find queue family indices to populate struct with
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
		// find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}

			if (indices.isComplete()) {
				break;
			}

			i++;
		}
		return indices;
	}
}