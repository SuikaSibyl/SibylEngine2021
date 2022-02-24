module;
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IDescriptorSetLayout.VK;
import RHI.IDescriptorSetLayout;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IDescriptorSetLayoutVK :public IDescriptorSetLayout
		{
		public:
			IDescriptorSetLayoutVK(DescriptorSetLayoutDesc const& desc, ILogicalDeviceVK* logical_device);
			virtual ~IDescriptorSetLayoutVK();

			auto getVkDescriptorSetLayout() noexcept -> VkDescriptorSetLayout*;

		private:
			VkDescriptorSetLayout layout;
			ILogicalDeviceVK* logicalDevice;
		};
	}
}
