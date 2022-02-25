module;
#include <vulkan/vulkan.h>
#include <utility>
module RHI.ITexture.VK;
import Core.Log;
import Core.MemoryManager;
import RHI.ITexture;
import RHI.IResource.VK;
import RHI.ILogicalDevice.VK;
import RHI.ITextureView;
import RHI.ITextureView.VK;
import RHI.IBuffer;
import RHI.IBuffer.VK;

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

	auto createVkImage(
		uint32_t const& width, 
		uint32_t const& height,
		VkImage* texture_image,
		VkDeviceMemory* device_memory,
		ILogicalDeviceVK* logical_device) noexcept -> void
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = static_cast<uint32_t>(width);
		imageInfo.extent.height = static_cast<uint32_t>(height);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.flags = 0; // Optional

		if (vkCreateImage(logical_device->getDeviceHandle(), &imageInfo, nullptr, texture_image) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to create image!");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(logical_device->getDeviceHandle(), *texture_image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = logical_device->getPhysicalDeviceVk()->
			findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

		if (vkAllocateMemory(logical_device->getDeviceHandle(), &allocInfo, nullptr, device_memory) != VK_SUCCESS) {
			SE_CORE_ERROR("failed to allocate image memory!");
		}

		vkBindImageMemory(logical_device->getDeviceHandle(), *texture_image, *device_memory, 0);
	}

	ITextureVK::ITextureVK(Image* image, ILogicalDeviceVK* _logical_device)
		:logicalDevice(_logical_device)
	{
		// create staging buffer
		BufferDesc stagingBufferDesc =
		{
			(unsigned int)image->getSize(),
			(BufferUsageFlags)BufferUsageFlagBits::TRANSFER_SRC_BIT,
			BufferShareMode::EXCLUSIVE,
			(uint32_t)MemoryPropertyFlagBits::HOST_VISIBLE_BIT | (uint32_t)MemoryPropertyFlagBits::HOST_COHERENT_BIT
		};
		IBufferVK stagingBuffer(stagingBufferDesc, logicalDevice);

		// copy memory
		void* data;
		vkMapMemory(logicalDevice->getDeviceHandle(), *(stagingBuffer.getVkDeviceMemory()), 0, stagingBuffer.getSize(), 0, &data);
		memcpy(data, image->getData(), (unsigned int)image->getSize());
		vkUnmapMemory(logicalDevice->getDeviceHandle(), *(stagingBuffer.getVkDeviceMemory()));



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
		resource = std::move(_texture.resource);
		logicalDevice = _texture.logicalDevice;
		externalImage = _texture.externalImage;

		_texture.image = nullptr;
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
			vkDestroyImage(logicalDevice->getDeviceHandle(), image, nullptr);
		}
		if (deviceMemory && !externalImage)
		{
			vkFreeMemory(logicalDevice->getDeviceHandle(), deviceMemory, nullptr);
		}
	}
}
