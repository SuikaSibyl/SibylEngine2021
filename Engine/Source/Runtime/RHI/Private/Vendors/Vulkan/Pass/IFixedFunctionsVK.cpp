module;
#include <vulkan/vulkan.h>
#include <vector>
module RHI.IFixedFunctions.VK;
import RHI.IFixedFunctions;
import RHI.IEnum.VK;

namespace SIByL::RHI
{
	auto IVertexLayoutVK::getVkInputState() noexcept -> VkPipelineVertexInputStateCreateInfo*
	{
		return &vertexInputInfo;
	}

	auto IVertexLayoutVK::createVkInputState() noexcept -> void
	{
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertexInputInfo.vertexBindingDescriptionCount = 0;
		vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
		vertexInputInfo.vertexAttributeDescriptionCount = 0;
		vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional
	}

	auto IInputAssemblyVK::createVkInputAssembly() noexcept -> void
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = getVkTopology(topologyKind);
		inputAssembly.primitiveRestartEnable = VK_FALSE;
	}

	auto IInputAssemblyVK::getVkInputAssembly() noexcept -> VkPipelineInputAssemblyStateCreateInfo*
	{
		return &inputAssembly;
	}

	auto createVkViewport(float width, float height) noexcept -> VkViewport
	{
		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = width;
		viewport.height = height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;
		return viewport;
	}

	auto createVkScissor(VkExtent2D* extend) noexcept -> VkRect2D
	{
		VkRect2D scissor{};
		scissor.offset = {0, 0};
		scissor.extent = *extend;
		return scissor;
	}

	auto IViewportsScissorsVK::getVkPipelineViewportStateCreateInfo() noexcept -> VkPipelineViewportStateCreateInfo*
	{
		return &viewportState;
	}

	auto IViewportsScissorsVK::createVkPipelineViewportStateCreateInfo
	(float width, float height, VkExtent2D* extend) noexcept -> void
	{
		viewport = createVkViewport(width, height);
		scissor = createVkScissor(extend);

		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.pViewports = &viewport;
		viewportState.scissorCount = 1;
		viewportState.pScissors = &scissor;
	}

	auto createRasterizerState() noexcept -> VkPipelineRasterizationStateCreateInfo
	{
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		// VK_POLYGON_MODE_FILL: fill the area of the polygon with fragments
		// VK_POLYGON_MODE_LINE: polygon edges are drawn as lines
		// VK_POLYGON_MODE_POINT : polygon vertices are drawn as points
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; // Optional
		rasterizer.depthBiasClamp = 0.0f; // Optional
		rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

		return rasterizer;
	}

	auto IRasterizerVK::getVkPipelineRasterizationStateCreateInfo() noexcept
		-> VkPipelineRasterizationStateCreateInfo*
	{
		return &rasterizer;
	}

	auto IRasterizerVK::createRasterizerStateInfo() noexcept -> void
	{
		rasterizer = createRasterizerState();
	}

	auto IMultisamplingVK::getVkPipelineMultisampleStateCreateInfo() noexcept
		-> VkPipelineMultisampleStateCreateInfo*
	{
		return &multisampling;
	}

	auto IMultisamplingVK::createMultisampingInfo() noexcept -> void
	{
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f; // Optional
		multisampling.pSampleMask = nullptr; // Optional
		multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
		multisampling.alphaToOneEnable = VK_FALSE; // Optional
	}

	auto IDepthStencilVK::getVkPipelineDepthStencilStateCreateInfo() noexcept
		-> VkPipelineDepthStencilStateCreateInfo*
	{
		return &depthStencil;
	}

	auto IColorBlendingVK::getVkPipelineColorBlendAttachmentState() noexcept
		-> VkPipelineColorBlendAttachmentState*
	{
		return &colorBlendAttachment;
	}

	auto IColorBlendingVK::getVkPipelineColorBlendStateCreateInfo() noexcept
		-> VkPipelineColorBlendStateCreateInfo*
	{
		return &colorBlending;
	}

	auto IColorBlendingVK::createColorBlendObjects() noexcept -> void
	{
		// VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional
		
		// VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f; // Optional
		colorBlending.blendConstants[1] = 0.0f; // Optional
		colorBlending.blendConstants[2] = 0.0f; // Optional
		colorBlending.blendConstants[3] = 0.0f; // Optional
	}

	auto IDynamicStateVK::createDynamicState() noexcept -> void
	{
		dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_LINE_WIDTH
		};

		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = dynamicStates.size();
		dynamicState.pDynamicStates = dynamicStates.data();
	}
}
