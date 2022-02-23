module;
#include <vector>
#include <cstdint>
#include <vulkan/vulkan.h>
export module RHI.IEnum.VK;
import RHI.IEnum;

namespace SIByL::RHI
{
	export inline auto getVkTopology(TopologyKind type) noexcept -> VkPrimitiveTopology;
	export inline auto getVkShaderStage(ShaderStage stage) noexcept -> VkShaderStageFlagBits;
	export inline auto getVkPolygonMode(PolygonMode mode) noexcept -> VkPolygonMode;
	export inline auto getVkCullMode(CullMode mode) noexcept -> VkCullModeFlagBits;
	export inline auto getVkBlendOperator(BlendOperator mode) noexcept -> VkBlendOp;
	export inline auto getVkBlendFactor(BlendFactor mode) noexcept -> VkBlendFactor;
	export inline auto getVkDynamicState(PipelineState state) noexcept -> VkDynamicState;
	export inline auto getVkSampleCount(SampleCount count) noexcept -> VkSampleCountFlagBits;
	export inline auto getVKFormat(ResourceFormat format) noexcept -> VkFormat;
	export inline auto getVkDataFormat(DataType datatype) noexcept -> VkFormat;
}