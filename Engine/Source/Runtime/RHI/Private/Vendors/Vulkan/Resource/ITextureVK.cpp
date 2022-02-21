module;
#include <vulkan/vulkan.h>
#include <utility>
module RHI.ITexture.VK;
import Core.Log;
import RHI.ITexture;
import RHI.IResource.VK;
import RHI.ILogicalDevice.VK;
import RHI.ITextureView;
import RHI.ITextureView.VK;
import Core.MemoryManager;

namespace SIByL::RHI
{
	auto createImageViews(TextureViewDesc const& desc, IResourceVK* resource, VkImage* image, ILogicalDeviceVK* logical_device, VkImageView* image_view) noexcept -> void
	{
		VkImageViewCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		createInfo.image = *image;
		createInfo.viewType = resource->getVKImageViewType();
		createInfo.format = resource->getVKFormat();
		// The components field allows you to swizzle the color channels around
		createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		// The subresourceRange field describes what the image's purpose is and 
		// which part of the image should be accessed. 
		// Our images will be used as color targets without any mipmapping levels or multiple layers.
		createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		createInfo.subresourceRange.baseMipLevel = 0;
		createInfo.subresourceRange.levelCount = 1;
		createInfo.subresourceRange.baseArrayLayer = 0;
		createInfo.subresourceRange.layerCount = 1;
;
		if (vkCreateImageView(logical_device->getDeviceHandle(), &createInfo, nullptr, image_view) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to create image views!");
		}
	}

	ITextureVK::ITextureVK(VkImage _image, IResourceVK&& _resource, ILogicalDeviceVK* _logical_device)
		: image(_image)
		, resource(std::move(_resource))
		, logicalDevice(_logical_device)
		, externalImage(true)
	{}

	ITextureVK::ITextureVK(ITextureVK&& _texture)
	{
		image = _texture.image;
		imageView = _texture.imageView;

		_texture.image = nullptr;
		_texture.imageView = nullptr;
	}

	auto ITextureVK::createView(TextureViewDesc const& desc) noexcept -> MemScope<ITextureView>
	{
		MemScope<ITextureViewVK> view = MemNew<ITextureViewVK>(logicalDevice);
		createImageViews(desc, &resource, &image, logicalDevice, view->getpVkImageView());
		MemScope<ITextureView> general_view = MemCast<ITextureView>(view);
		return general_view;
	}

	ITextureVK::~ITextureVK()
	{
		if (image && !externalImage)
		{
			// destroy image
		}
		if (imageView)
		{
			vkDestroyImageView(logicalDevice->getDeviceHandle(), imageView, nullptr);
		}
	}
}
