module;
#include <vulkan/vulkan.h>
#include <vector>
export module RHI.GraphicContext.VK;
import RHI.GraphicContext;
import Core.Window;
import Core.Window.GLFW;

namespace SIByL
{
	namespace RHI
	{
		export class IGraphicContextVK :public IGraphicContext
		{
		public:
			IGraphicContextVK();
			static auto getVKInstance() -> VkInstance&;
			// SObject
			virtual auto initialize() -> bool;
			virtual auto destroy() -> bool;
			// IGraphicContext
			virtual auto attachWindow(IWindow* window) noexcept -> void override;

		private:
			static VkInstance instance;

		public:
			auto hasValidationLayers() -> bool;
			auto getSurface() noexcept -> VkSurfaceKHR&;
			auto getAttachedWindow() noexcept -> IWindowGLFW*;

			auto createInstance() -> void;
			auto checkExtension() -> void;
			auto checkValidationLayerSupport() -> bool;
			auto getRequiredExtensions() -> std::vector<const char*>;
			auto setupDebugMessenger() -> void;

			VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
			VkDebugUtilsMessengerEXT debugMessenger;
			VkSurfaceKHR surface;
			IWindowGLFW* windowAttached;
		};
	}
}