module;
#include <cstdint>
#include <vector>
export module RHI.IPipelineLayout;
import Core.SObject;
import RHI.IDescriptorSetLayout;

namespace SIByL
{
	namespace RHI
	{
		// IPipelineLayouts declares uniform values layout
		// In Vulkan/DirectX12 it should be specified during pipeline creation.
		// 
		// ╭──────────────┬─────────────────────╮
		// │  Vulkan	  │   VkPipelineLayout  │
		// │  DirectX 12  │   RootSignature**   │
		// │  OpenGL      │                     │
		// ╰──────────────┴─────────────────────╯

		export struct PipelineLayoutDesc
		{
			std::vector<IDescriptorSetLayout*> layouts;
		};

		export class IPipelineLayout :public SObject
		{
		public:
			IPipelineLayout() = default;
			IPipelineLayout(IPipelineLayout&&) = default;
			virtual ~IPipelineLayout() = default;

		};
	}
}
