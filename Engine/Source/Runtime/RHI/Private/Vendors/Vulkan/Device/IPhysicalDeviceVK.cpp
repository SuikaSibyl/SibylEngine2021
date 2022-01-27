module;
#include <vulkan/vulkan.h>
#include <vector>
#include <set>
#include <string>
module RHI.IPhysicalDevice.VK;
import Core.Log;
import RHI.GraphicContext;
import RHI.GraphicContext.VK;

namespace SIByL::RHI
{
#ifdef _DEBUG
	const bool enableValidationLayers = true;
#else
	const bool enableValidationLayers = false;
#endif

	IPhysicalDeviceVK::IPhysicalDeviceVK(IGraphicContext* context)
	{
		graphicContext = dynamic_cast<IGraphicContextVK*>(context);
	}

	auto IPhysicalDeviceVK::initialize() -> bool
	{
		enableDebugLayer = enableValidationLayers;
		queryAllPhysicalDevice();
		return true;
	}

	bool IPhysicalDeviceVK::checkDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	auto IPhysicalDeviceVK::querySwapChainSupport(VkPhysicalDevice device)->SwapChainSupportDetails
	{
		SwapChainSupportDetails details;
		// query basic surface capabilities
		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, graphicContext->getSurface(), &details.capabilities);
		// query the supported surface formats
		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, graphicContext->getSurface(), &formatCount, nullptr);
		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, graphicContext->getSurface(), &formatCount, details.formats.data());
		}
		// query the supported presentation modes
		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, graphicContext->getSurface(), &presentModeCount, nullptr);
		if (presentModeCount != 0) {
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, graphicContext->getSurface(), &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	auto IPhysicalDeviceVK::isDeviceSuitable(VkPhysicalDevice device) -> bool
	{
		QueueFamilyIndices indices = findQueueFamilies(device);
		// check extension supports
		bool extensionsSupported = checkDeviceExtensionSupport(device);
		// check swapchain support
		bool swapChainAdequate = false;
		if (extensionsSupported) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}

		return indices.isComplete() && extensionsSupported && swapChainAdequate;
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

	auto IPhysicalDeviceVK::getDeviceExtensions() noexcept -> std::vector<const char*> const&
	{
		return deviceExtensions;
	}

	auto IPhysicalDeviceVK::findQueueFamilies(VkPhysicalDevice device)->QueueFamilyIndices
	{
		QueueFamilyIndices indices;
		// Logic to find queue family indices to populate struct with
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
		// find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			// check graphic support
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}
			// check present support
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, graphicContext->getSurface(), &presentSupport);
			if (presentSupport) {
				indices.presentFamily = i;
			}
			// check support completeness
			if (indices.isComplete()) {
				break;
			}

			i++;
		}
		return indices;
	}

	auto IPhysicalDeviceVK::findQueueFamilies() -> QueueFamilyIndices
	{
		return findQueueFamilies(physicalDevice);
	}

	auto IPhysicalDeviceVK::querySwapChainSupport()->SwapChainSupportDetails
	{
		return querySwapChainSupport(physicalDevice);
	}

	auto IPhysicalDeviceVK::getGraphicContext()->IGraphicContextVK*
	{
		return graphicContext;
	}

}