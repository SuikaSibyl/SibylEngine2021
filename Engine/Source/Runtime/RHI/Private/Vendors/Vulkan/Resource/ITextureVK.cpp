module;
#include <vulkan/vulkan.h>
#include <utility>
module RHI.ITexture.VK;
import Core.Log;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.IEnum.VK;
import RHI.ITexture;
import RHI.IResource.VK;
import RHI.ILogicalDevice.VK;
import RHI.ITextureView;
import RHI.ITextureView.VK;
import RHI.IBuffer;
import RHI.IBuffer.VK;
import RHI.IDeviceGlobal;
import RHI.ICommandPool;
import RHI.ICommandBuffer;
import RHI.ICommandPool.VK;
import RHI.ICommandBuffer.VK;
import RHI.IFactory;
import RHI.IBarrier;
import RHI.IMemoryBarrier;

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
		TextureDesc const& desc,
		VkImage* texture_image,
		VkDeviceMemory* device_memory,
		ILogicalDeviceVK* logical_device) noexcept -> void
	{
		// create actual image
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = getVkImageType(desc.type);
		imageInfo.extent.width = static_cast<uint32_t>(desc.width);
		imageInfo.extent.height = static_cast<uint32_t>(desc.height);
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.format = getVKFormat(desc.format);
		imageInfo.tiling = getVkImageTiling(desc.tiling);
		imageInfo.initialLayout = getVkImageLayout(desc.layout);
		imageInfo.usage = getVkImageUsageFlags(desc.usages);
		imageInfo.sharingMode = getVkBufferShareMode(desc.shareMode);
		imageInfo.samples = getVkSampleCount(desc.sampleCount);
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

		// copy memory to staging buffer
		void* data;
		vkMapMemory(logicalDevice->getDeviceHandle(), *(stagingBuffer.getVkDeviceMemory()), 0, stagingBuffer.getSize(), 0, &data);
		memcpy(data, image->getData(), (unsigned int)image->getSize());
		vkUnmapMemory(logicalDevice->getDeviceHandle(), *(stagingBuffer.getVkDeviceMemory()));

		// create actual image
		desc =
		{
			ResourceType::Texture2D,
			ResourceFormat::FORMAT_R8G8B8A8_SRGB,
			ImageTiling::OPTIMAL,
			ImageUsageFlags(ImageUsageFlagBits::TRANSFER_DST_BIT) | ImageUsageFlags(ImageUsageFlagBits::SAMPLED_BIT),
			BufferShareMode::EXCLUSIVE,
			SampleCount::COUNT_1_BIT,
			ImageLayout::UNDEFINED,
			image->getWidth(),
			image->getHeight()
		};
		createVkImage(desc, &this->image, &deviceMemory, logicalDevice);

		// transition to TRANSFER_DEST_OPTIMAL
		transitionImageLayout(ImageLayout::UNDEFINED, ImageLayout::TRANSFER_DST_OPTIMAL);
		// buffer image copy
		PerDeviceGlobal* global = DeviceToGlobal::getGlobal((ILogicalDevice*)logicalDevice);
		MemScope<IBufferImageCopy> bic = global->getResourceFactory()->createBufferImageCopy({
			0,0,0,(uint32_t)ImageAspectFlagBits::COLOR_BIT,0,0,1,0,0,0,desc.width, desc.height, 1
			});
		ICommandPool* transientPool = global->getTransientCommandPool();
		MemScope<ICommandBuffer> commandbuffer = global->getResourceFactory()->createCommandBuffer(transientPool);
		commandbuffer->beginRecording((uint32_t)CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT);
		commandbuffer->cmdCopyBufferToImage(&stagingBuffer, this, bic.get());
		// end recording
		commandbuffer->endRecording();
		commandbuffer->submit();
		logicalDevice->waitIdle();
		// transition to
		transitionImageLayout(ImageLayout::TRANSFER_DST_OPTIMAL, ImageLayout::SHADER_READ_ONLY_OPTIMAL);
	}

	auto hasDepthComponent(ResourceFormat format) noexcept -> bool
	{
		return (format == ResourceFormat::FORMAT_D32_SFLOAT_S8_UINT || format ==  ResourceFormat::FORMAT_D32_SFLOAT || format == ResourceFormat::FORMAT_D24_UNORM_S8_UINT);
	}

	ITextureVK::ITextureVK(TextureDesc const& _desc, ILogicalDeviceVK* _logical_device)
		: logicalDevice(_logical_device)
		, desc(_desc)
	{
		createVkImage(desc, &this->image, &deviceMemory, logicalDevice);
		// layout changement
		if (hasDepthComponent(desc.format))
			transitionImageLayout(ImageLayout::UNDEFINED, ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA);
	}

	ITextureVK::ITextureVK(VkImage _image, IResourceVK&& _resource, TextureDesc const& desc, ILogicalDeviceVK* _logical_device)
		: image(_image)
		, resource(std::move(_resource))
		, logicalDevice(_logical_device)
		, externalImage(true)
		, desc(desc)
	{}

	ITextureVK::ITextureVK(ITextureVK&& _texture)
	{
		image = _texture.image;
		resource = std::move(_texture.resource);
		logicalDevice = _texture.logicalDevice;
		externalImage = _texture.externalImage;
		desc = _texture.desc;

		_texture.image = nullptr;
	}
	
	auto hasStencilComponent(ResourceFormat format) noexcept -> bool
	{
		return (format == ResourceFormat::FORMAT_D32_SFLOAT_S8_UINT || format == ResourceFormat::FORMAT_D24_UNORM_S8_UINT);
	}

	auto ITextureVK::transitionImageLayout(ImageLayout old_layout, ImageLayout new_layout) noexcept -> void
	{
		// get stage mask
		AccessFlags srcAccessMask = 0;
		AccessFlags dstAccessMask = 0;

		PipelineStageFlags sourceStage = 0;
		PipelineStageFlags destinationStage = 0;

		if (old_layout == ImageLayout::UNDEFINED && new_layout == ImageLayout::TRANSFER_DST_OPTIMAL) {
			srcAccessMask = 0;
			dstAccessMask = (uint32_t)AccessFlagBits::TRANSFER_WRITE_BIT;

			sourceStage = (uint32_t)PipelineStageFlagBits::TOP_OF_PIPE_BIT;
			destinationStage = (uint32_t)PipelineStageFlagBits::TRANSFER_BIT;
		}
		else if (old_layout == ImageLayout::TRANSFER_DST_OPTIMAL && new_layout == ImageLayout::SHADER_READ_ONLY_OPTIMAL) {
			srcAccessMask = (uint32_t)AccessFlagBits::TRANSFER_WRITE_BIT;
			dstAccessMask = (uint32_t)AccessFlagBits::SHADER_READ_BIT;

			sourceStage = (uint32_t)PipelineStageFlagBits::TRANSFER_BIT;
			destinationStage = (uint32_t)PipelineStageFlagBits::FRAGMENT_SHADER_BIT;
		}
		else if (old_layout == ImageLayout::UNDEFINED && new_layout == ImageLayout::SHADER_READ_ONLY_OPTIMAL) {
			srcAccessMask = 0;
			dstAccessMask = (uint32_t)AccessFlagBits::SHADER_READ_BIT;

			sourceStage = (uint32_t)PipelineStageFlagBits::TOP_OF_PIPE_BIT;
			destinationStage = (uint32_t)PipelineStageFlagBits::FRAGMENT_SHADER_BIT;
		}
		else if (old_layout == ImageLayout::UNDEFINED && new_layout == ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA) {
			srcAccessMask = 0;
			dstAccessMask = (uint32_t)AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT | (uint32_t)AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

			sourceStage = (uint32_t)PipelineStageFlagBits::TOP_OF_PIPE_BIT;
			destinationStage = (uint32_t)PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT;
		}
		else if (old_layout == ImageLayout::UNDEFINED && new_layout == ImageLayout::PRESENT_SRC) {
			srcAccessMask = 0;
			dstAccessMask = (uint32_t)AccessFlagBits::TRANSFER_WRITE_BIT;

			sourceStage = (uint32_t)PipelineStageFlagBits::TOP_OF_PIPE_BIT;
			destinationStage = (uint32_t)PipelineStageFlagBits::TRANSFER_BIT;
		}
		else if (old_layout == ImageLayout::UNDEFINED && new_layout == ImageLayout::GENERAL) {
			srcAccessMask = 0;
			dstAccessMask = (uint32_t)AccessFlagBits::SHADER_WRITE_BIT | (uint32_t)AccessFlagBits::MEMORY_WRITE_BIT;

			sourceStage = (uint32_t)PipelineStageFlagBits::TOP_OF_PIPE_BIT;
			destinationStage = (uint32_t)PipelineStageFlagBits::COMPUTE_SHADER_BIT;
		}
		else {
			SE_CORE_ERROR("VULKAN :: Image layout transition :: unsupported layout transition!");
		}

		ImageAspectFlags aspectMask = 0;

		// get right subresource aspect
		if (new_layout == ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA) {
			aspectMask = (ImageAspectFlags)ImageAspectFlagBits::DEPTH_BIT;

			if (hasStencilComponent(desc.format)) {
				aspectMask |= (ImageAspectFlags)ImageAspectFlagBits::STENCIL_BIT;
			}
		}
		else {
			aspectMask = (ImageAspectFlags)ImageAspectFlagBits::COLOR_BIT;
		}

		// begin recording
		PerDeviceGlobal* global = DeviceToGlobal::getGlobal((ILogicalDevice*)logicalDevice);
		ICommandPool* transientPool = global->getTransientCommandPool();
		MemScope<ICommandBuffer> commandbuffer = global->getResourceFactory()->createCommandBuffer(transientPool);
		commandbuffer->beginRecording((uint32_t)CommandBufferUsageFlagBits::ONE_TIME_SUBMIT_BIT);

		MemScope<IImageMemoryBarrier> image_memory_barrier = global->getResourceFactory()->createImageMemoryBarrier({
			this, //ITexture* image;
			ImageSubresourceRange{
				aspectMask,
				0,
				1,
				0,
				1
			},//ImageSubresourceRange subresourceRange;
			srcAccessMask, //AccessFlags srcAccessMask;
			dstAccessMask, //AccessFlags dstAccessMask;
			old_layout, // old Layout
			new_layout // new Layout
			});

		MemScope<IBarrier> barrier = global->getResourceFactory()->createBarrier({
			sourceStage,//srcStageMask
			destinationStage,//dstStageMask
			0,
			{},
			{},
			{image_memory_barrier.get()}
			});

		commandbuffer->cmdPipelineBarrier(barrier.get());

		// end recording
		commandbuffer->endRecording();
		commandbuffer->submit();
		logicalDevice->waitIdle();
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

	inline auto getVkImageAspectFlags(ImageAspectFlags _flag) noexcept -> VkImageAspectFlags
	{
		uint32_t flags{};
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::COLOR_BIT, VK_IMAGE_ASPECT_COLOR_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::DEPTH_BIT, VK_IMAGE_ASPECT_DEPTH_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::STENCIL_BIT, VK_IMAGE_ASPECT_STENCIL_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::METADATA_BIT, VK_IMAGE_ASPECT_METADATA_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::PLANE_0_BIT, VK_IMAGE_ASPECT_PLANE_0_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::PLANE_1_BIT, VK_IMAGE_ASPECT_PLANE_1_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::PLANE_2_BIT, VK_IMAGE_ASPECT_PLANE_2_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::MEMORY_PLANE_0_BIT, VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::MEMORY_PLANE_1_BIT, VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::MEMORY_PLANE_2_BIT, VK_IMAGE_ASPECT_MEMORY_PLANE_2_BIT_EXT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::ImageAspectFlagBits::MEMORY_PLANE_3_BIT, VK_IMAGE_ASPECT_MEMORY_PLANE_3_BIT_EXT, flags);
		return (VkImageAspectFlags)flags;
	}

	IBufferImageCopyVK::IBufferImageCopyVK(BufferImageCopyDesc const& desc)
	{
		copy.bufferOffset = desc.bufferOffset;
		copy.bufferRowLength = desc.bufferRowLength;
		copy.bufferImageHeight = desc.bufferImageHeight;

		copy.imageSubresource.aspectMask = getVkImageAspectFlags(desc.aspectMask);
		copy.imageSubresource.mipLevel = desc.mipLevel;
		copy.imageSubresource.baseArrayLayer = desc.baseArrayLayer;
		copy.imageSubresource.layerCount = desc.layerCount;

		copy.imageOffset = { desc.imageOffsetX, desc.imageOffsetY, desc.imageOffsetZ };
		copy.imageExtent = { desc.imageExtentX, desc.imageExtentY, desc.imageExtentZ };
	}
}
