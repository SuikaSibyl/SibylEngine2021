module;
#include <cstdint>
#include <vector>
export module RHI.IDescriptorSetLayout;
import RHI.IEnum;
import RHI.ISampler;
namespace SIByL
{
	namespace RHI
	{
		// Descriptor Set Layout: specify types of resource.
		// ╭──────────────┬────────────────────────────╮
		// │  Vulkan	  │   VkDescriptorSetLayout    │
		// │  DirectX 12  │                            │
		// │  OpenGL      │                            │
		// ╰──────────────┴────────────────────────────╯

		export struct DescriptorSetLayoutPerBindingDesc
		{
			uint32_t binding;
			uint32_t count;
			DescriptorType type;
			ShaderStageFlags stages;
			ISampler* immutableSampler;
		};

		export struct DescriptorSetLayoutDesc
		{
			std::vector<DescriptorSetLayoutPerBindingDesc> perBindingDesc;
		};

		export class IDescriptorSetLayout
		{
		public:
			IDescriptorSetLayout() = default;
			virtual ~IDescriptorSetLayout() = default;

		private:
		};
	}
}
