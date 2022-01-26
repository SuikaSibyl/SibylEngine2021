module;
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
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

			struct QueueFamilyIndices {
				std::optional<uint32_t> graphicsFamily;

				bool isComplete() {
					return graphicsFamily.has_value();
				}
			};
			auto findQueueFamilies()->QueueFamilyIndices;

		public:
			auto getPhysicalDevice() noexcept -> VkPhysicalDevice&;

		private:
			bool enableDebugLayer;
			VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
			std::vector<VkPhysicalDevice> devices;

		private:
			auto queryAllPhysicalDevice() noexcept -> void;
			auto isDeviceSuitable(VkPhysicalDevice device) -> bool;
			auto rateDeviceSuitability(VkPhysicalDevice device) -> int;
		};
	}
}