module;
#include <cstdint>
export module RHI.IDescriptorSet;
import Core.SObject;
import RHI.IDescriptorSetLayout;
import RHI.IDescriptorPool;
import RHI.IUniformBuffer;
import RHI.ITextureView;
import RHI.ISampler;

namespace SIByL
{
	namespace RHI
	{
		// Descriptor Set: specify the actual resource for descriptor.
		// ╭──────────────┬───────────────────────╮
		// │  Vulkan	  │   VkDescriptorSet     │
		// │  DirectX 12  │                       │
		// │  OpenGL      │                       │
		// ╰──────────────┴───────────────────────╯
		export struct DescriptorSetDesc
		{
			IDescriptorPool* descriptorPool;
			IDescriptorSetLayout* layout;
		};

		export class IDescriptorSet
		{
		public:
			IDescriptorSet() = default;
			virtual ~IDescriptorSet() = default;

			virtual auto update(IUniformBuffer* uniform_buffer, uint32_t const& binding, uint32_t const& array_element) noexcept -> void = 0;
			virtual auto update(ITextureView* texture_view, ISampler* sampler, uint32_t const& binding, uint32_t const& array_element) noexcept -> void = 0;

		private:
		};
	}
}
