module;
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
export module RHI.GraphicContext.VK;
import RHI.GraphicContext;

namespace SIByL
{
	namespace RHI
	{
		export struct QueueFamilyIndicesVK {
			std::optional<uint32_t> graphicsFamily;

			bool isComplete() {
				return graphicsFamily.has_value();
			}
		};

		export class IGraphicContextVK
		{
		public:
			auto findQueueFamilies() -> QueueFamilyIndicesVK;
			auto hasValidationLayers() -> bool;

		public:
			auto initVulkan() -> void;
			auto createInstance() -> void;
			auto checkExtension() -> void;
			auto checkValidationLayerSupport() -> bool;
			auto getRequiredExtensions() -> std::vector<const char*>;
			auto cleanUp() -> void;
			auto setupDebugMessenger() -> void;
			auto isDeviceSuitable(VkPhysicalDevice device) -> bool;
			auto pickPhysicalDevice() -> void;
			auto findQueueFamilies(VkPhysicalDevice device) ->QueueFamilyIndicesVK;
			
			VkInstance instance;
			VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
			VkDebugUtilsMessengerEXT debugMessenger;
		};
	}
}