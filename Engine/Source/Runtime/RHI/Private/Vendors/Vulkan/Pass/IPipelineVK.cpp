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

		IPipelineVK::IPipelineVK(ComputePipelineDesc const& _desc, ILogicalDeviceVK* logical_device)
			: logicalDevice(logical_device)
		{
			createVkComputePipeline(_desc);
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

		auto getVkPipelineDepthStencilStateCreateInfo() noexcept -> VkPipelineDepthStencilStateCreateInfo
		{
			VkPipelineDepthStencilStateCreateInfo depthStencil{};
			depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
			depthStencil.depthTestEnable = VK_TRUE;
			depthStencil.depthWriteEnable = VK_TRUE;
			depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
			depthStencil.depthBoundsTestEnable = VK_FALSE;
			depthStencil.minDepthBounds = 0.0f; // Optional
			depthStencil.maxDepthBounds = 1.0f; // Optional
			depthStencil.stencilTestEnable = VK_FALSE;
			depthStencil.front = {}; // Optional
			depthStencil.back = {}; // Optional
			return depthStencil;
		}

		auto IPipelineVK::createVkPipeline() noexcept -> void
		{
			std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
			shaderStages.resize(desc.shaders.size());
			for (int i = 0; i < desc.shaders.size(); i++)
			{
				shaderStages[i] = *(static_cast<IShaderVK*>(desc.shaders[i]))->getVkShaderStageCreateInfo();
			}

			VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info = getVkPipelineDepthStencilStateCreateInfo();

			VkGraphicsPipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			pipelineInfo.stageCount = shaderStages.size();
			pipelineInfo.pStages = shaderStages.data();
			pipelineInfo.pVertexInputState = static_cast<IVertexLayoutVK*>(desc.vertexLayout)->getVkInputState();
			pipelineInfo.pInputAssemblyState = static_cast<IInputAssemblyVK*>(desc.inputAssembly)->getVkInputAssembly();
			pipelineInfo.pViewportState = static_cast<IViewportsScissorsVK*>(desc.viewportsScissors)->getVkPipelineViewportStateCreateInfo();
			pipelineInfo.pRasterizationState = static_cast<IRasterizerVK*>(desc.rasterizer)->getVkPipelineRasterizationStateCreateInfo();
			pipelineInfo.pMultisampleState = static_cast<IMultisamplingVK*>(desc.multisampling)->getVkPipelineMultisampleStateCreateInfo();
			pipelineInfo.pDepthStencilState = &depth_stencil_create_info; // TODO :: Optional
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

		auto IPipelineVK::createVkComputePipeline(ComputePipelineDesc const& desc) noexcept -> void
		{
			VkComputePipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			pipelineInfo.stage.module = (static_cast<IShaderVK*>(desc.shader))->getVkShaderModule();
			pipelineInfo.stage.pName = "main";
			pipelineInfo.layout = *(static_cast<IPipelineLayoutVK*>(desc.pipelineLayout)->getVkPipelineLayout());

			if (vkCreateComputePipelines(logicalDevice->getDeviceHandle(), VK_NULL_HANDLE,
				1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
				SE_CORE_ERROR("VULKAN :: failed to create graphics pipeline!");
			}
		}
	}
}
