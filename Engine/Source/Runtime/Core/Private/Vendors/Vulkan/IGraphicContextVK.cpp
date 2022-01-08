module;
#include <vector>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>;
import Core.Log;
module Core.GraphicContext.VK;
import Core.GraphicContext;

namespace SIByL
{
	inline namespace Core
	{
		auto IGraphicContextVK::createInstance() -> void
		{
			// optional
			// but it may provide some useful information to the driver in order to optimize our specific application
			VkApplicationInfo appInfo{};
			appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			appInfo.pApplicationName = "Application";
			appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.pEngineName = "No Engine";
			appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.apiVersion = VK_API_VERSION_1_0;

			// not optional
			// Tells the Vulkan driver which global extensions and validation layers we want to use.
			// Global here means that they apply to the entire program and not a specific device,
			VkInstanceCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			createInfo.pApplicationInfo = &appInfo;
			
			// specify the desired global extensions
			uint32_t glfwExtensionCount = 0;
			const char** glfwExtensions;

			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

			createInfo.enabledExtensionCount = glfwExtensionCount;
			createInfo.ppEnabledExtensionNames = glfwExtensions;

			// determine the global validation layers to enable
			createInfo.enabledLayerCount = 0;
			
			VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

			if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
				SE_CORE_ERROR("failed to create instance!");
			}
		}

		auto IGraphicContextVK::checkExtension() -> void
		{
			uint32_t extensionCount = 0;
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

			std::vector<VkExtensionProperties> extensions(extensionCount);
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

			SE_CORE_INFO("Vulkan: available extensions:");

			for (const auto& extension : extensions) {
				SE_CORE_INFO("\t {0}", extension.extensionName);
			}
		}

		auto IGraphicContextVK::cleanUp() -> void
		{
			vkDestroyInstance(instance, nullptr);
		}

	}
}