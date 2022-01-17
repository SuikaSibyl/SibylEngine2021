module;
#include <vulkan/vulkan.h>
#include <vector>
export module RHI.GraphicContext.VK;
import RHI.GraphicContext;

namespace SIByL
{
	namespace RHI
	{
		export class IGraphicContextVK :public IGraphicContext
		{
		public:
			static auto getVKInstance() -> VkInstance&;

			virtual auto initialize() -> bool;
			virtual auto destroy() -> bool;

		private:
			static VkInstance instance;

		public:
			auto hasValidationLayers() -> bool;

		public:
			auto createInstance() -> void;
			auto checkExtension() -> void;
			auto checkValidationLayerSupport() -> bool;
			auto getRequiredExtensions() -> std::vector<const char*>;
			auto setupDebugMessenger() -> void;
			
			VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
			VkDebugUtilsMessengerEXT debugMessenger;
		};
	}
}