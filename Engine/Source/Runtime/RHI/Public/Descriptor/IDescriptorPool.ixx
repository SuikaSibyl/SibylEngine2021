module;
#include <cstdint>
#include <vector>
export module RHI.IDescriptorPool;
import Core.SObject;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		// Descriptor Pool: GPU Pool for allocating Descriptor Set.
		// ╭──────────────┬───────────────────────╮
		// │  Vulkan	  │   VkDescriptorPool    │
		// │  DirectX 12  │                       │
		// │  OpenGL      │                       │
		// ╰──────────────┴───────────────────────╯

		export struct DescriptorPoolDesc
		{
			std::vector<std::pair<DescriptorType, uint32_t>> typeAndCount;
			uint32_t max_sets;
		};

		export class IDescriptorPool
		{
		public:
			IDescriptorPool() = default;
			virtual ~IDescriptorPool() = default;
		};
	}
}
