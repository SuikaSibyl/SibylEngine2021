module;
#include <vulkan/vulkan.h>
#include <vector>
export module Core.GraphicContext.VK;
import Core.GraphicContext;

namespace SIByL
{
	inline namespace Core
	{
		export class IGraphicContextVK
		{
		public:
			auto initVulkan() -> void;
			auto createInstance() -> void;
			auto checkExtension() -> void;
			auto checkValidationLayerSupport() -> bool;
			auto getRequiredExtensions() -> std::vector<const char*>;
			auto cleanUp() -> void;
			auto setupDebugMessenger() -> void;

			VkInstance instance;
			VkDebugUtilsMessengerEXT debugMessenger;
		};
	}
}