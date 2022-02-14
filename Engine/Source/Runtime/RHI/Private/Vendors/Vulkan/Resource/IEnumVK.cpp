module;
#include <cstdint>
#include <vulkan/vulkan.h>
module RHI.IEnum.VK;
import RHI.IEnum;
import Core.Log;

namespace SIByL::RHI
{
	inline auto getVkTopology(TopologyKind type) noexcept -> VkPrimitiveTopology
	{
		switch (type)
		{
		case SIByL::RHI::TopologyKind::TriangleStrip:
			return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
			break;
		case SIByL::RHI::TopologyKind::TriangleList:
			return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
			break;
		case SIByL::RHI::TopologyKind::LineStrip:
			return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
			break;
		case SIByL::RHI::TopologyKind::LineList:
			return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			break;
		case SIByL::RHI::TopologyKind::PointList:
			return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
			break;
		default:
			break;
		}

#ifdef _DEBUG
		SE_CORE_ERROR("VULKAN :: Wrong Topology Kind");
#endif // _DEBUG

		return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	}

	inline auto getVkShaderStage(ShaderStage stage) noexcept -> VkShaderStageFlagBits
	{
		switch (stage)
		{
		case SIByL::RHI::ShaderStage::MESH:
			return VK_SHADER_STAGE_MESH_BIT_NV;
			break;
		case SIByL::RHI::ShaderStage::COMPUTE:
			return VK_SHADER_STAGE_COMPUTE_BIT;
			break;
		case SIByL::RHI::ShaderStage::TESSELLATION:
			return VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
			break;
		case SIByL::RHI::ShaderStage::GEOMETRY:
			return VK_SHADER_STAGE_GEOMETRY_BIT;
			break;
		case SIByL::RHI::ShaderStage::FRAGMENT:
			return VK_SHADER_STAGE_FRAGMENT_BIT;
			break;
		case SIByL::RHI::ShaderStage::VERTEX:
			return VK_SHADER_STAGE_VERTEX_BIT;
			break;
		default:
			break;
		}
#ifdef _DEBUG
		SE_CORE_ERROR("VULKAN :: Wrong Shader Stage");
#endif // _DEBUG'
		return VK_SHADER_STAGE_ALL;
	}
}