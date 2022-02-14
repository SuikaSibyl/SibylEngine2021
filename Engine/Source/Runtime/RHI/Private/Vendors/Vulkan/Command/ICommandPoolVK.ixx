module;
#include <vulkan/vulkan.h>
export module RHI.ICommandPool.VK;
import RHI.ICommandPool;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		export class ICommandPoolVK :public ICommandPool
		{
		public:
			ICommandPoolVK() = default;
			ICommandPoolVK(ICommandPoolVK const&) = delete;
			ICommandPoolVK(ICommandPoolVK&&) = delete;
			virtual ~ICommandPoolVK();

			auto getVkCommandPool() noexcept -> VkCommandPool*;
		private:
			VkCommandPool commandPool;
			ILogicalDeviceVK* logicalDevice;
			auto createVkCommandPool() noexcept -> void;
		};
	}
}
