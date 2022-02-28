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
		auto createImageView(VkImage image, VkFormat format, VkImageView* imageView, ILogicalDeviceVK* logical_device) noexcept -> void
		{
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = format;
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
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
			createImageView(*((ITextureVK*)texture)->getVkImage(), getVKFormat(((ITextureVK*)texture)->getDescription().format), &imageView, logicalDevice);
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
