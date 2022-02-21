module;
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
export module RHI.ILogicalDevice.VK;
import RHI.ILogicalDevice;
import RHI.IPhysicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class ILogicalDeviceVK :public ILogicalDevice
		{
		public:
			ILogicalDeviceVK(IPhysicalDeviceVK* physicalDevice);
			virtual ~ILogicalDeviceVK() = default;

			virtual auto initialize() -> bool;
			virtual auto destroy() -> bool;
			virtual auto getPhysicalDevice() noexcept -> IPhysicalDevice* override;

			auto getDeviceHandle() noexcept -> VkDevice&;
			auto getPhysicalDeviceVk() noexcept -> IPhysicalDeviceVK*;

		private:
			IPhysicalDeviceVK* physicalDevice;
			VkDevice device;
			VkQueue graphicsQueue;
			VkQueue presentQueue;

		private:
			auto createLogicalDevice(IPhysicalDeviceVK* physicalDevice) noexcept -> void;
		};
	}
}