module;
#include <vulkan/vulkan.h>
module RHI.IResource.VK;
import RHI.IResource;
import RHI.IEnum;

namespace SIByL::RHI
{
	auto IResourceVK::getVKFormat() noexcept -> VkFormat
	{
		switch (format)
		{
		case ResourceFormat::FORMAT_B8G8R8A8_RGB:
			return VK_FORMAT_B8G8R8A8_UINT;
			break;
		case ResourceFormat::FORMAT_B8G8R8A8_SRGB:
			return VK_FORMAT_B8G8R8A8_SRGB;
			break;
		case ResourceFormat::FORMAT_R8G8B8A8_SRGB:
			return VK_FORMAT_R8G8B8A8_SRGB;
			break;
		default:
			break;
		}
		return VK_FORMAT_UNDEFINED;
	}

	auto IResourceVK::getVKImageViewType() noexcept -> VkImageViewType
	{
		switch (type)
		{
		case ResourceType::Buffer:
			return VK_IMAGE_VIEW_TYPE_MAX_ENUM;
			break;
		case ResourceType::Texture1D:
			return VK_IMAGE_VIEW_TYPE_1D;
			break;
		case ResourceType::Texture2D:
			return VK_IMAGE_VIEW_TYPE_2D;
			break;
		case ResourceType::Texture3D:
			return VK_IMAGE_VIEW_TYPE_3D;
			break;
		case ResourceType::TextureCube:
			return VK_IMAGE_VIEW_TYPE_CUBE;
			break;
		case ResourceType::Texture2DMultisample:
			return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
			break;
		default:
			break;
		}
		return VK_IMAGE_VIEW_TYPE_MAX_ENUM;
	}

	auto IResourceVK::VKFormat2IFormat(VkFormat _format) noexcept -> ResourceFormat
	{
		ResourceFormat format;
		switch (_format)
		{
		case VK_FORMAT_R8G8B8A8_UINT:
			format = ResourceFormat::FORMAT_B8G8R8A8_RGB;
			break;
		case VK_FORMAT_B8G8R8A8_SRGB:
			format = ResourceFormat::FORMAT_B8G8R8A8_SRGB;
			break;
		default:
			break;
		}
		return format;
	}

	auto IResourceVK::VKImageViewType2IType(VkImageViewType _type) noexcept -> ResourceType
	{
		ResourceType type;
		switch (_type)
		{
		case VK_IMAGE_VIEW_TYPE_1D:
			type = ResourceType::Texture1D;
			break;
		case VK_IMAGE_VIEW_TYPE_2D:
			type = ResourceType::Texture2D;
			break;
		case VK_IMAGE_VIEW_TYPE_3D:
			type = ResourceType::Texture3D;
			break;
		case VK_IMAGE_VIEW_TYPE_CUBE:
			type = ResourceType::TextureCube;
			break;
		case VK_IMAGE_VIEW_TYPE_2D_ARRAY:
			type = ResourceType::Texture2DMultisample;
			break;
		default:
			break;
		}
		return type;
	}

	auto IResourceVK::setVKFormat(VkFormat _format) noexcept -> bool
	{
		format = VKFormat2IFormat(_format);
		return true;
	}

	auto IResourceVK::setVKImageViewType(VkImageViewType _type) noexcept -> bool
	{
		type = VKImageViewType2IType(_type);
		return true;
	}
}