module;
#include <vector>
#include <vulkan/vulkan.h>
#include <cstdint> // Necessary for UINT32_MAX
#include <algorithm> // Necessary for std::clamp
#include <utility>
module RHI.ISwapChain.VK;
import Core.Log;
import RHI.ISwapChain;
import RHI.IPhysicalDevice.VK;
import RHI.ITexture;
import RHI.ITexture.VK;
import RHI.IResource.VK;
import RHI.ITextureView;
import RHI.ITextureView.VK;

namespace SIByL::RHI
{
	ISwapChainVK::ISwapChainVK(ILogicalDeviceVK* logical_device)
	{
		physicalDevice = logical_device->getPhysicalDeviceVk();
		logicalDevice = logical_device;
		windowAttached = physicalDevice->getGraphicContextVK()->getAttachedWindow();
		createSwapChain();
	}

	ISwapChainVK::~ISwapChainVK()
	{
		if (swapChain)
			vkDestroySwapchainKHR(logicalDevice->getDeviceHandle(), swapChain, nullptr);
	}
	
	auto ISwapChainVK::getExtend() noexcept -> Extend
	{
		return { swapChainExtent.width, swapChainExtent.height };
	}

	auto ISwapChainVK::createSwapChain() noexcept -> void
	{
		IPhysicalDeviceVK::SwapChainSupportDetails swapChainSupport = physicalDevice->querySwapChainSupport();

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		// Desc
		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = physicalDevice->getGraphicContextVK()->getSurface();
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
		
		IPhysicalDeviceVK::QueueFamilyIndices indices = physicalDevice->findQueueFamilies();
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		if (indices.graphicsFamily.value() != indices.presentFamily.value()) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0; // Optional
			createInfo.pQueueFamilyIndices = nullptr; // Optional
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;
		
		if (vkCreateSwapchainKHR(logicalDevice->getDeviceHandle(), &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
			SE_CORE_ERROR("failed to create swap chain!");
		}

		std::vector<VkImage> images;
		vkGetSwapchainImagesKHR(logicalDevice->getDeviceHandle(), swapChain, &imageCount, nullptr);
		images.resize(imageCount);
		vkGetSwapchainImagesKHR(logicalDevice->getDeviceHandle(), swapChain, &imageCount, images.data());
		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;

		for (uint32_t i = 0; i < imageCount; i++)
		{
			IResourceVK resource;
			resource.setVKFormat(swapChainImageFormat);
			resource.setVKImageViewType(VK_IMAGE_VIEW_TYPE_2D);
			ITextureVK texture(images[i], std::move(resource), logicalDevice);
			swapchainViews.emplace_back(texture.createView({}));
			swapChainTextures.emplace_back(std::move(texture));
		}
	}

	// There are three types of settings to determine:
	//	1. Surface format(color depth)
	//	2. Presentation mode(conditions for "swapping" images to the screen)
	//	3. Swap extent(resolution of images in swap chain)

	auto ISwapChainVK::chooseSwapSurfaceFormat(std::vector<VkSurfaceFormatKHR> const& availableFormats) noexcept -> VkSurfaceFormatKHR
	{
		// For the formats. find VK_FORMAT_B8G8R8A8_SRGB
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	auto ISwapChainVK::chooseSwapPresentMode(std::vector<VkPresentModeKHR> const& availablePresentModes) noexcept -> VkPresentModeKHR
	{
		// VK_PRESENT_MODE_MAILBOX_KHR allows us to avoid tearing while still maintaining a fairly low latency
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	auto ISwapChainVK::chooseSwapExtent(VkSurfaceCapabilitiesKHR const& capabilities) noexcept -> VkExtent2D
	{
		if (capabilities.currentExtent.width != UINT32_MAX) {
			return capabilities.currentExtent;
		}
		else {
			uint32_t width, height;
			windowAttached -> getFramebufferSize(width, height);

			VkExtent2D actualExtent = {
				width,
				height,
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}
}
