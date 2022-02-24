module;
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IPipelineLayout.VK;
import Core.Log;
import RHI.IPipelineLayout;
import RHI.IDescriptorSetLayout;
import RHI.IDescriptorSetLayout.VK;
import RHI.ILogicalDevice.VK;

namespace SIByL::RHI
{
	IPipelineLayoutVK::IPipelineLayoutVK(PipelineLayoutDesc const& desc, ILogicalDeviceVK* logical_device)
		:logicalDevice(logical_device)
	{
		createPipelineLayout(desc);
	}

	IPipelineLayoutVK::~IPipelineLayoutVK()
	{
		if (pipelineLayout && logicalDevice)
			vkDestroyPipelineLayout(logicalDevice->getDeviceHandle(), pipelineLayout, nullptr);
	}

	auto IPipelineLayoutVK::getVkPipelineLayout() noexcept -> VkPipelineLayout*
	{
		return &pipelineLayout;
	}

	auto IPipelineLayoutVK::createPipelineLayout(PipelineLayoutDesc const& desc) noexcept -> void
	{
		std::vector<VkDescriptorSetLayout> layouts(desc.layouts.size());
		for (int i = 0; i < layouts.size(); i++)
		{
			layouts[i] = *((IDescriptorSetLayoutVK*)desc.layouts[i])->getVkDescriptorSetLayout();
		}

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = desc.layouts.size(); // Optional
		pipelineLayoutInfo.pSetLayouts = layouts.data(); // Optional
		pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
		pipelineLayoutInfo.pPushConstantRanges = nullptr; // Optional

		if (vkCreatePipelineLayout(logicalDevice->getDeviceHandle(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to create pipeline layout!");
		}
	}

}
