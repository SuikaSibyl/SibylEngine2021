module;
#include <cstdint>
#include <vulkan/vulkan.h>
export module RHI.IFence.VK;
import RHI.IFence;
import Core.SObject;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IFenceVK :public IFence
		{
		public:
			IFenceVK(ILogicalDeviceVK* logical_device);
			virtual ~IFenceVK();
			virtual auto wait() noexcept -> void override;
			virtual auto reset() noexcept -> void override;

			auto getVkFence() noexcept -> VkFence*;

		private:
			ILogicalDeviceVK* logicalDevice;
			VkFence vkFence;
		};
	}
}
