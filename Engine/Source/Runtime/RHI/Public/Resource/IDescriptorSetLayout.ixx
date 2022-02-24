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
			DescriptorSetLayoutDesc(std::initializer_list<DescriptorSetLayoutPerBindingDesc> const& per_binding_desc)
				:perBindingDesc(per_binding_desc)
			{}

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
