module;
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IPipeline.VK;
import Core.Log;
import RHI.IPipeline;
import RHI.IResource;
import RHI.IShader;
import RHI.IShader.VK;
import RHI.IFixedFunctions;
import RHI.IFixedFunctions.VK;
import RHI.IPipelineLayout;
import RHI.IPipelineLayout.VK;
import RHI.IRenderPass;
import RHI.IRenderPass.VK;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		IPipelineVK::IPipelineVK(PipelineDesc const& _desc, ILogicalDeviceVK* logical_device)
			: logicalDevice(logical_device)
			, desc(_desc)
		{
			createVkPipeline();
		}

		IPipelineVK::~IPipelineVK()
		{
			if (graphicsPipeline)
				vkDestroyPipeline(logicalDevice->getDeviceHandle(), graphicsPipeline, nullptr);
		}

		auto IPipelineVK::getVkPipeline() noexcept -> VkPipeline*
		{
			return &graphicsPipeline;
		}

		auto IPipelineVK::createVkPipeline() noexcept -> void
		{
			std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
			shaderStages.resize(desc.shaders.size());
			for (int i = 0; i < desc.shaders.size(); i++)
			{
				shaderStages[i] = *(static_cast<IShaderVK*>(desc.shaders[i]))->getVkShaderStageCreateInfo();
			}

			VkGraphicsPipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			pipelineInfo.stageCount = shaderStages.size();
			pipelineInfo.pStages = shaderStages.data();
			pipelineInfo.pVertexInputState = static_cast<IVertexLayoutVK*>(desc.vertexLayout)->getVkInputState();
			pipelineInfo.pInputAssemblyState = static_cast<IInputAssemblyVK*>(desc.inputAssembly)->getVkInputAssembly();
			pipelineInfo.pViewportState = static_cast<IViewportsScissorsVK*>(desc.viewportsScissors)->getVkPipelineViewportStateCreateInfo();
			pipelineInfo.pRasterizationState = static_cast<IRasterizerVK*>(desc.rasterizer)->getVkPipelineRasterizationStateCreateInfo();
			pipelineInfo.pMultisampleState = static_cast<IMultisamplingVK*>(desc.multisampling)->getVkPipelineMultisampleStateCreateInfo();
			pipelineInfo.pDepthStencilState = nullptr; // Optional
			pipelineInfo.pColorBlendState = static_cast<IColorBlendingVK*>(desc.colorBlending)->getVkPipelineColorBlendStateCreateInfo();
			pipelineInfo.pDynamicState = nullptr; // Optional
			pipelineInfo.layout = *(static_cast<IPipelineLayoutVK*>(desc.pipelineLayout)->getVkPipelineLayout());
			pipelineInfo.renderPass = *(static_cast<IRenderPassVK*>(desc.renderPass)->getRenderPass());
			pipelineInfo.subpass = 0;
			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
			pipelineInfo.basePipelineIndex = -1; // Optional

			if (vkCreateGraphicsPipelines(logicalDevice->getDeviceHandle(), VK_NULL_HANDLE,
				1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
				SE_CORE_ERROR("VULKAN :: failed to create graphics pipeline!");
			}
		}
	}
}
