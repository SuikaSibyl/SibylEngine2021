module;
#include <vulkan/vulkan.h>
module RHI.IShader.VK;
import Core.SObject;
import Core.Log;
import RHI.IShader;
import RHI.IEnum;
import RHI.IEnum.VK;

namespace SIByL::RHI
{
	IShaderVK::~IShaderVK()
	{
		if (shaderModule)
			vkDestroyShaderModule(logicalDevice->getDeviceHandle(), shaderModule, nullptr);
	}

	auto IShaderVK::getVkShaderModule() noexcept -> VkShaderModule&
	{
		return shaderModule;
	}

	auto IShaderVK::createShaderModule(char const* code, size_t size) noexcept -> void
	{
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = size;
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code);

		if (vkCreateShaderModule(logicalDevice->getDeviceHandle(), &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to create shader module!");
		}
	}

	auto IShaderVK::getVkShaderStageCreateInfo() noexcept -> VkPipelineShaderStageCreateInfo*
	{
		return &shaderStageInfo;
	}

	auto IShaderVK::createVkShaderStageCreateInfo() noexcept -> void
	{
		shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStageInfo.stage = getVkShaderStage(stage);
		shaderStageInfo.module = shaderModule;
		shaderStageInfo.pName = "entryPointName";
		return;
	}
}