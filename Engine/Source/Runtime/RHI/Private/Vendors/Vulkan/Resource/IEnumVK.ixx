module;
#include <cstdint>
#include <vulkan/vulkan.h>
export module RHI.IEnum.VK;
import RHI.IEnum;

namespace SIByL::RHI
{
	export inline auto getVkTopology(TopologyKind type) noexcept -> VkPrimitiveTopology;
	export inline auto getVkShaderStage(ShaderStage stage) noexcept -> VkShaderStageFlagBits;
}