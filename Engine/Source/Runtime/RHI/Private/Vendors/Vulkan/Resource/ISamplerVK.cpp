module;
#include <vulkan/vulkan.h>
module RHI.ISampler.VK;
import Core.Log;
import RHI.ISampler;
import RHI.ILogicalDevice.VK;
import RHI.IPhysicalDevice.VK;

namespace SIByL::RHI
{
	auto decode(MipmapMode mipmapMode) noexcept -> VkSamplerMipmapMode
	{
		switch (mipmapMode)
		{
		case SIByL::RHI::MipmapMode::NEAREST:
			return VK_SAMPLER_MIPMAP_MODE_NEAREST;
			break;
		case SIByL::RHI::MipmapMode::LINEAR:
			return VK_SAMPLER_MIPMAP_MODE_LINEAR;
			break;
		default:
			break;
		}
	}

	auto decode(AddressMode addressMode) noexcept ->VkSamplerAddressMode
	{
		switch (addressMode)
		{
		case AddressMode::REPEAT:
			return VK_SAMPLER_ADDRESS_MODE_REPEAT;
			break;
		case AddressMode::CLAMP_TO_EDGE:
			return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
			break;
		}
	}

	void createTextureSampler(
		SamplerDesc const& desc,
		VkSampler* texture_sampler,
		ILogicalDeviceVK* logical_device
	) {
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = decode(desc.clampModeU);
		samplerInfo.addressModeV = decode(desc.clampModeV);
		samplerInfo.addressModeW = decode(desc.clampModeW);
		samplerInfo.anisotropyEnable = VK_TRUE;

		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(logical_device->getPhysicalDeviceVk()->getPhysicalDevice(), &properties);
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = decode(desc.mipmapMode);
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = (float)desc.minLod;
		samplerInfo.maxLod = (float)desc.maxLod;
		
		// optional extension
		VkSamplerReductionModeCreateInfoEXT createInfoReduction = {};
		createInfoReduction.sType = VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT;
		if (desc.extension == Extension::MIN_POOLING)
		{
			//add a extension struct to enable Min mode
			createInfoReduction.reductionMode = VK_SAMPLER_REDUCTION_MODE_MIN;
			samplerInfo.pNext = &createInfoReduction;
		}

		if (vkCreateSampler(logical_device->getDeviceHandle(), &samplerInfo, nullptr, texture_sampler) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to create texture sampler!");
		}
	}

	ISamplerVK::ISamplerVK(SamplerDesc const& desc, ILogicalDeviceVK* logical_device)
		: logicalDevice(logical_device)
	{
		createTextureSampler(desc, &textureSampler, logicalDevice);
	}

	ISamplerVK::~ISamplerVK()
	{
		if (textureSampler)
			vkDestroySampler(logicalDevice->getDeviceHandle(), textureSampler, nullptr);
	}
}