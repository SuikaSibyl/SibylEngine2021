module;
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IBarrier.VK;
import RHI.IBarrier;
import RHI.IEnum;

namespace SIByL::RHI
{
	auto getVkPipelineStageFlags(PipelineStageFlags _flag) noexcept -> VkPipelineStageFlags;
	auto getVkDependencyTypeFlags(DependencyTypeFlags _flag) noexcept -> VkDependencyFlags;

	IBarrierVK::IBarrierVK(BarrierDesc const& desc)
	{
		srcStageMask = getVkPipelineStageFlags(desc.srcStageMask);
		srcStageMask = getVkPipelineStageFlags(desc.dstStageMask);
		dependencyFlags = getVkDependencyTypeFlags(desc.dependencyType);

		// TODO
		// - memory barrier
		// - buffer memory barrier
	}

	auto IBarrierVK::getMemoryBarrierData() noexcept -> VkMemoryBarrier*
	{
		if (getMemoryBarrierCount())
			return memoryBarriers.data();
		else
			return nullptr;
	}

	auto IBarrierVK::getBufferMemoryBarrierData() noexcept -> VkBufferMemoryBarrier*
	{
		if (getMemoryBarrierCount())
			return bufferMemoryBarriers.data();
		else
			return nullptr;
	}

	auto IBarrierVK::getImageMemoryBarrierData() noexcept -> VkImageMemoryBarrier*
	{
		if (getMemoryBarrierCount())
			return imageMemoryBarriers.data();
		else
			return nullptr;
	}


	auto getVkPipelineStageFlags(PipelineStageFlags _flag) noexcept -> VkPipelineStageFlags
	{
		uint32_t flags{};
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::TESSELLATION_CONTROL_SHADER_BIT, VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::TESSELLATION_EVALUATION_SHADER_BIT, VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::GEOMETRY_SHADER_BIT, VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::HOST_BIT, VK_PIPELINE_STAGE_HOST_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::TRANSFORM_FEEDBACK_BIT_EXT, VK_PIPELINE_STAGE_TRANSFORM_FEEDBACK_BIT_EXT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::CONDITIONAL_RENDERING_BIT_EXT, VK_PIPELINE_STAGE_CONDITIONAL_RENDERING_BIT_EXT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::TASK_SHADER_BIT_NV, VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::MESH_SHADER_BIT_NV, VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::FRAGMENT_DENSITY_PROCESS_BIT, VK_PIPELINE_STAGE_FRAGMENT_DENSITY_PROCESS_BIT_EXT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::FRAGMENT_SHADING_RATE_ATTACHMENT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::PipelineStageFlagBits::COMMAND_PREPROCESS_BIT, VK_PIPELINE_STAGE_COMMAND_PREPROCESS_BIT_NV, flags);

		return (VkImageUsageFlags)flags;
	}

	auto getVkDependencyTypeFlags(DependencyTypeFlags _flag) noexcept -> VkDependencyFlags
	{
		uint32_t flags{};
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::DependencyTypeFlagBits::BY_REGION_BIT, VK_DEPENDENCY_BY_REGION_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::DependencyTypeFlagBits::VIEW_LOCAL_BIT, VK_DEPENDENCY_VIEW_LOCAL_BIT, flags);
		flagBitSwitch(_flag, (uint32_t)SIByL::RHI::DependencyTypeFlagBits::DEVICE_GROUP_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, flags);
		return (VkImageUsageFlags)flags;
	}
}