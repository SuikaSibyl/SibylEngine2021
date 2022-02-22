module;
#include <vulkan/vulkan.h>
#include <vector>
module RHI.IFixedFunctions.VK;
import RHI.IFixedFunctions;
import RHI.IEnum.VK;

namespace SIByL::RHI
{
	IVertexLayoutVK::IVertexLayoutVK()
	{
		createVkInputState();
	}

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

	IInputAssemblyVK::IInputAssemblyVK(TopologyKind topology_kind)
		:IInputAssembly(topology_kind)
	{
		createVkInputAssembly();
	}

	auto IInputAssemblyVK::createVkInputAssembly() noexcept -> void
	{
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

	IViewportsScissorsVK::IViewportsScissorsVK(
		unsigned int width_viewport,
		unsigned int height_viewport,
		unsigned int width_scissor,
		unsigned int height_scissor)
	{
		VkExtent2D extend{ width_scissor ,height_scissor };
		createVkPipelineViewportStateCreateInfo(width_viewport, height_viewport, &extend);
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

	auto createRasterizerState(RasterizerDesc const& desc) noexcept -> VkPipelineRasterizationStateCreateInfo
	{
		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		// VK_POLYGON_MODE_FILL: fill the area of the polygon with fragments
		// VK_POLYGON_MODE_LINE: polygon edges are drawn as lines
		// VK_POLYGON_MODE_POINT : polygon vertices are drawn as points
		rasterizer.polygonMode = getVkPolygonMode(desc.polygonMode);
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = getVkCullMode(desc.cullMode);
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f; // Optional
		rasterizer.depthBiasClamp = 0.0f; // Optional
		rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

		return rasterizer;
	}
	
	IRasterizerVK::IRasterizerVK(RasterizerDesc const& desc)
	{
		createRasterizerStateInfo(desc);
	}

	auto IRasterizerVK::getVkPipelineRasterizationStateCreateInfo() noexcept
		-> VkPipelineRasterizationStateCreateInfo*
	{
		return &rasterizer;
	}

	auto IRasterizerVK::createRasterizerStateInfo(RasterizerDesc const& desc) noexcept -> void
	{
		rasterizer = createRasterizerState(desc);
	}

	IMultisamplingVK::IMultisamplingVK(MultiSampleDesc const& desc)
	{
		createMultisampingInfo(desc);
	}

	auto IMultisamplingVK::getVkPipelineMultisampleStateCreateInfo() noexcept
		-> VkPipelineMultisampleStateCreateInfo*
	{
		return &multisampling;
	}

	auto IMultisamplingVK::createMultisampingInfo(MultiSampleDesc const& desc) noexcept -> void
	{
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f; // Optional
		multisampling.pSampleMask = nullptr; // Optional
		multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
		multisampling.alphaToOneEnable = VK_FALSE; // Optional
	}

	IDepthStencilVK::IDepthStencilVK(DepthStencilDesc const& desc)
	{
		
	}

	auto IDepthStencilVK::getVkPipelineDepthStencilStateCreateInfo() noexcept
		-> VkPipelineDepthStencilStateCreateInfo*
	{
		if (!initialized) return nullptr;
		return &depthStencil;
	}

	IColorBlendingVK::IColorBlendingVK(ColorBlendingDesc const& desc)
	{
		createColorBlendObjects(desc);
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

	auto IColorBlendingVK::createColorBlendObjects(ColorBlendingDesc const& desc) noexcept -> void
	{
		// VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = desc.blendEnable ? VK_TRUE : VK_FALSE;
		colorBlendAttachment.srcColorBlendFactor = getVkBlendFactor(desc.srcColorFactor); // Optional
		colorBlendAttachment.dstColorBlendFactor = getVkBlendFactor(desc.dstColorFactor); // Optional
		colorBlendAttachment.colorBlendOp = getVkBlendOperator(desc.colorOperator); // Optional
		colorBlendAttachment.srcAlphaBlendFactor = getVkBlendFactor(desc.srcAlphaFactor); // Optional
		colorBlendAttachment.dstAlphaBlendFactor = getVkBlendFactor(desc.dstAlphaFactor); // Optional
		colorBlendAttachment.alphaBlendOp = getVkBlendOperator(desc.alphaOperator); // Optional
		
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

	IDynamicStateVK::IDynamicStateVK(std::vector<PipelineState> const& states)
	{
		createDynamicState(states);
	}

	auto IDynamicStateVK::createDynamicState(std::vector<PipelineState> const& states) noexcept -> void
	{
		for (int i = 0; i < states.size(); i++)
		{
			dynamicStates.emplace_back(getVkDynamicState(states[i]));
		}

		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = dynamicStates.size();
		dynamicState.pDynamicStates = dynamicStates.data();
	}
}
