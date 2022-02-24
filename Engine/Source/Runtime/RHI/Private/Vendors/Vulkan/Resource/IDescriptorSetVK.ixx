module;
#include <vulkan/vulkan.h>
export module RHI.IDescriptorSet.VK;
import RHI.IDescriptorSet;
import RHI.ILogicalDevice.VK;
import RHI.IDescriptorSetLayout;
import RHI.IDescriptorPool;
import RHI.IDescriptorPool.VK;
import RHI.IUniformBuffer;

namespace SIByL
{
	namespace RHI
	{
		export class IDescriptorSetVK :public IDescriptorSet
		{
		public:
			IDescriptorSetVK(DescriptorSetDesc const& desc, ILogicalDeviceVK* logical_device);
			virtual ~IDescriptorSetVK();

			virtual auto update(IUniformBuffer* uniform_buffer, uint32_t const& binding, uint32_t const& array_element) noexcept -> void override;
			auto getVkDescriptorSet() noexcept -> VkDescriptorSet*;

		private:
			VkDescriptorSet set;
			IDescriptorPoolVK* descriptorPool;
			ILogicalDeviceVK* logicalDevice;
		};
	}
}
