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
			ISemaphoreVK();
			ISemaphoreVK(ISemaphoreVK const&) = delete;
			ISemaphoreVK(ISemaphoreVK&&) = delete;
			virtual ~ISemaphoreVK() = default;

		private:
			ILogicalDeviceVK* logicalDevice;
			VkSemaphore semaphore;

			auto createSemaphores() noexcept -> void;
		};
	}
}
