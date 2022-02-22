module;
#include <vulkan/vulkan.h>
export module RHI.ISemaphore.VK;
import RHI.ISemaphore;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		export class ISemaphoreVK :public ISemaphore
		{
		public:
			ISemaphoreVK(ILogicalDeviceVK* logical_device);
			virtual ~ISemaphoreVK();

			auto getVkSemaphore() noexcept -> VkSemaphore*;

		private:
			ILogicalDeviceVK* logicalDevice;
			VkSemaphore semaphore;

			auto createSemaphores() noexcept -> void;
		};
	}
}
