module;
#include <vulkan/vulkan.h>
export module RHI.IDescriptorPool.VK;
import RHI.IDescriptorPool;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IDescriptorPoolVK :public IDescriptorPool
		{
		public:
			IDescriptorPoolVK(DescriptorPoolDesc const& desc, ILogicalDeviceVK* logical_device);
			virtual ~IDescriptorPoolVK();

			auto getVkDescriptorPool() noexcept -> VkDescriptorPool*;

		private:
			VkDescriptorPool descriptorPool;
			ILogicalDeviceVK* logicalDevice;
		};
	}
}
