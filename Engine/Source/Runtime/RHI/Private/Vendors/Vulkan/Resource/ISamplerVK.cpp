module;
#include <vulkan/vulkan.h>
module RHI.ISampler.VK;
import Core.Log;
import RHI.ISampler;
import RHI.ILogicalDevice.VK;
import RHI.IPhysicalDevice.VK;

namespace SIByL::RHI
{
	void createTextureSampler(
		VkSampler* texture_sampler,
		ILogicalDeviceVK* logical_device
	) {
		VkSamplerCreateInfo samplerInfo{};
		samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		samplerInfo.magFilter = VK_FILTER_LINEAR;
		samplerInfo.minFilter = VK_FILTER_LINEAR;
		samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		samplerInfo.anisotropyEnable = VK_TRUE;

		VkPhysicalDeviceProperties properties{};
		vkGetPhysicalDeviceProperties(logical_device->getPhysicalDeviceVk()->getPhysicalDevice(), &properties);
		samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
		samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		samplerInfo.unnormalizedCoordinates = VK_FALSE;
		samplerInfo.compareEnable = VK_FALSE;
		samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		samplerInfo.mipLodBias = 0.0f;
		samplerInfo.minLod = 0.0f;
		samplerInfo.maxLod = 0.0f;

		if (vkCreateSampler(logical_device->getDeviceHandle(), &samplerInfo, nullptr, texture_sampler) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to create texture sampler!");
		}
	}

	ISamplerVK::ISamplerVK(SamplerDesc const& desc, ILogicalDeviceVK* logical_device)
		: logicalDevice(logical_device)
	{
		createTextureSampler(&textureSampler, logicalDevice);
	}

	ISamplerVK::~ISamplerVK()
	{
		if (textureSampler)
			vkDestroySampler(logicalDevice->getDeviceHandle(), textureSampler, nullptr);
	}
}