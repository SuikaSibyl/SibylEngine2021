module;
#include <vulkan/vulkan.h>
export module RHI.IPhysicalDevice.VK;
import RHI.IPhysicalDevice;

namespace SIByL
{
	namespace RHI
	{
		export class IPhysicalDeviceVK :public IPhysicalDevice
		{
		public:
			virtual ~IPhysicalDeviceVK() = default;

			virtual auto initialize() -> bool;
			virtual auto isDebugLayerEnabled() noexcept -> bool;

		private:
			bool enableDebugLayer;

			VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		};
	}
}