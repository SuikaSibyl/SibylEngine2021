module;
#include <vulkan/vulkan.h>
module RHI.ITextureView.VK;
import Core.SObject;
import Core.Log;
import RHI.ITextureView;
import RHI.ILogicalDevice.VK;
import RHI.ITexture;
import RHI.ITexture.VK;
import RHI.IEnum;
import RHI.IEnum.VK;

namespace SIByL
{
	namespace RHI
	{
		auto createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectMask, VkImageView* imageView, ILogicalDeviceVK* logical_device) noexcept -> void
		{
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = format;
			viewInfo.subresourceRange.aspectMask = aspectMask;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;

			if (vkCreateImageView(logical_device->getDeviceHandle(), &viewInfo, nullptr, imageView) != VK_SUCCESS) {
				SE_CORE_ERROR("VULKAN :: failed to create texture image view!");
			}
		}

		ITextureViewVK::ITextureViewVK(ILogicalDeviceVK* _logical_device)
			:logicalDevice(_logical_device)
		{}

		ITextureViewVK::ITextureViewVK(ITexture* texture, ILogicalDeviceVK* _logical_device)
			:logicalDevice(_logical_device)
		{
			ResourceFormat format = ((ITextureVK*)texture)->getDescription().format;
			VkImageAspectFlags accessMask = {};
			switch (format)
			{
			case SIByL::RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT:
			case SIByL::RHI::ResourceFormat::FORMAT_D32_SFLOAT_S8_UINT:
				accessMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
				break;
			case SIByL::RHI::ResourceFormat::FORMAT_D32_SFLOAT:
				accessMask = VK_IMAGE_ASPECT_DEPTH_BIT;
				break;
			case SIByL::RHI::ResourceFormat::FORMAT_R8G8B8A8_SRGB:
			case SIByL::RHI::ResourceFormat::FORMAT_B8G8R8A8_SRGB:
			case SIByL::RHI::ResourceFormat::FORMAT_B8G8R8A8_RGB:
			case SIByL::RHI::ResourceFormat::FORMAT_R32G32B32A32_SFLOAT:
				accessMask = VK_IMAGE_ASPECT_COLOR_BIT;
				break;
			default:
				SE_CORE_ERROR("VULKAN :: Image layout transition :: unsupported layout transition!");
				break;
			}

			createImageView(*((ITextureVK*)texture)->getVkImage(), getVKFormat(((ITextureVK*)texture)->getDescription().format), accessMask, &imageView, logicalDevice);
		}

		ITextureViewVK::~ITextureViewVK()
		{
			if (imageView)
			{
				vkDestroyImageView(logicalDevice->getDeviceHandle(), imageView, nullptr);
			}
		}

		auto ITextureViewVK::getpVkImageView() noexcept -> VkImageView*
		{
			return &imageView;
		}
	}
}
