module;
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
#include <string>
export module RHI.IPhysicalDevice.VK;
import RHI.IPhysicalDevice;
import RHI.GraphicContext;
import RHI.GraphicContext.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IPhysicalDeviceVK :public IPhysicalDevice
		{
		public:
			IPhysicalDeviceVK(IGraphicContext* context);
			virtual ~IPhysicalDeviceVK() = default;
			// IPhysicalDevice
			virtual auto initialize() -> bool;
			virtual auto isDebugLayerEnabled() noexcept -> bool;
			virtual auto getGraphicContext() noexcept -> IGraphicContext* override;
			virtual auto getTimestampPeriod() noexcept -> float override;

		public:
			struct QueueFamilyIndices {
				std::optional<uint32_t> graphicsFamily;
				std::optional<uint32_t> presentFamily;
				std::optional<uint32_t> computeFamily;

				bool isComplete() {
					return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value() && computeFamily.has_value();
				}
			};

			struct SwapChainSupportDetails {
				VkSurfaceCapabilitiesKHR capabilities;
				std::vector<VkSurfaceFormatKHR> formats;
				std::vector<VkPresentModeKHR> presentModes;
			};

			auto getPhysicalDevice() noexcept -> VkPhysicalDevice&;
			auto findQueueFamilies()->QueueFamilyIndices;
			auto getDeviceExtensions() noexcept -> std::vector<const char*> const&;
			auto querySwapChainSupport() -> SwapChainSupportDetails;
			auto getGraphicContextVK()->IGraphicContextVK*;
			auto findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) noexcept -> uint32_t;
			auto checkSubgroupProperties() noexcept -> void;

		private:
			bool enableDebugLayer;
			IGraphicContextVK* graphicContext;
			VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
			std::vector<VkPhysicalDevice> devices;
			std::vector<const char*> deviceExtensions = {
				VK_KHR_SWAPCHAIN_EXTENSION_NAME
			};

		private:
			auto queryAllPhysicalDevice() noexcept -> void;
			auto isDeviceSuitable(VkPhysicalDevice device, std::string& device_diagnosis) -> bool;
			auto rateDeviceSuitability(VkPhysicalDevice device) -> int;
			auto findQueueFamilies(VkPhysicalDevice device)->QueueFamilyIndices;
			auto checkDeviceExtensionSupport(VkPhysicalDevice device, std::string& device_diagnosis) -> bool;
			auto querySwapChainSupport(VkPhysicalDevice device) -> SwapChainSupportDetails;
			auto findSupportedFormat(std::vector<VkFormat> const& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) noexcept -> VkFormat;

		private:
			float timestampPeriod = 0.0f;
		};
	}
}