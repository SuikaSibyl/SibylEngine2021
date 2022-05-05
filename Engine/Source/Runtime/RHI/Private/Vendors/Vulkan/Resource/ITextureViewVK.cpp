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
		auto createImageView(VkImage image, VkFormat format, ImageSubresourceRange const& range, VkImageView* imageView, ILogicalDeviceVK* logical_device, TextureDesc const& desc) noexcept -> void
		{
			VkImageViewCreateInfo viewInfo{};
			viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			viewInfo.image = image;
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = format;
			viewInfo.subresourceRange.aspectMask = range.aspectMask;
			viewInfo.subresourceRange.baseMipLevel = range.baseMipLevel;
			viewInfo.subresourceRange.levelCount = range.levelCount;
			viewInfo.subresourceRange.baseArrayLayer = range.baseArrayLayer;
			viewInfo.subresourceRange.layerCount = range.layerCount;

			if (vkCreateImageView(logical_device->getDeviceHandle(), &viewInfo, nullptr, imageView) != VK_SUCCESS) {
				SE_CORE_ERROR("VULKAN :: failed to create texture image view!");
			}
		}

		ITextureViewVK::ITextureViewVK(ILogicalDeviceVK* _logical_device)
			:logicalDevice(_logical_device)
		{}

		ITextureViewVK::ITextureViewVK(ITexture* texture, ILogicalDeviceVK* _logical_device, ImageUsageFlags extra_usages)
			:logicalDevice(_logical_device)
		{
			ResourceFormat format = ((ITextureVK*)texture)->getDescription().format;
			ImageAspectFlags accessMask = {};
			switch (format)
			{
			case SIByL::RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT:
			case SIByL::RHI::ResourceFormat::FORMAT_D32_SFLOAT_S8_UINT:
				accessMask = (uint32_t)ImageAspectFlagBits::DEPTH_BIT | (uint32_t)ImageAspectFlagBits::STENCIL_BIT;
				break;
			case SIByL::RHI::ResourceFormat::FORMAT_D32_SFLOAT:
				accessMask = (uint32_t)ImageAspectFlagBits::DEPTH_BIT;
				break;
			case SIByL::RHI::ResourceFormat::FORMAT_R8G8B8A8_SRGB:
			case SIByL::RHI::ResourceFormat::FORMAT_B8G8R8A8_SRGB:
			case SIByL::RHI::ResourceFormat::FORMAT_B8G8R8A8_RGB:
			case SIByL::RHI::ResourceFormat::FORMAT_R8G8B8A8_UNORM:
			case SIByL::RHI::ResourceFormat::FORMAT_R32G32B32A32_SFLOAT:
			case SIByL::RHI::ResourceFormat::FORMAT_B10G11R11_UFLOAT_PACK32:
			case SIByL::RHI::ResourceFormat::FORMAT_R32_UINT:
			case SIByL::RHI::ResourceFormat::FORMAT_R32_SFLOAT:
				accessMask = (uint32_t)ImageAspectFlagBits::COLOR_BIT;
				break;
			default:
				SE_CORE_ERROR("VULKAN :: Image layout transition :: unsupported layout transition!");
				break;
			}

			ImageSubresourceRange range = {
				accessMask,
				0,
				((ITextureVK*)texture)->getDescription().mipLevels,
				0,
				1
			};

			createImageView(*((ITextureVK*)texture)->getVkImage(), getVKFormat(((ITextureVK*)texture)->getDescription().format), range, &imageView, logicalDevice, texture->getDescription());
		}

		ITextureViewVK::ITextureViewVK(ITexture* texture, ILogicalDeviceVK* _logical_device, ImageSubresourceRange const& range)
			:logicalDevice(_logical_device)
		{
			createImageView(*((ITextureVK*)texture)->getVkImage(), getVKFormat(((ITextureVK*)texture)->getDescription().format), range, &imageView, logicalDevice, texture->getDescription());
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
