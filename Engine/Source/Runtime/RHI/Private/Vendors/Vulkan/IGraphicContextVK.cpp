module;
#include <vector>
#include <optional>
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>;
import Core.Log;
module RHI.GraphicContext.VK;
import RHI.GraphicContext;

namespace SIByL
{
	namespace RHI
	{
		const std::vector<const char*> validationLayers = {
			"VK_LAYER_KHRONOS_validation"
		};

#ifdef _DEBUG
		const bool enableValidationLayers = true;
#else
		const bool enableValidationLayers = false;
#endif

		static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
			VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
			VkDebugUtilsMessageTypeFlagsEXT messageType,
			const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
			void* pUserData) {

			switch (messageSeverity)
			{
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
				//SE_CORE_TRACE("VULKAN :: VALIDATION :: {0}", pCallbackData->pMessage);
				break;
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
				SE_CORE_INFO("VULKAN :: VALIDATION :: {0}", pCallbackData->pMessage);
				break;
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
				SE_CORE_WARN("VULKAN :: VALIDATION :: {0}", pCallbackData->pMessage);
				break;
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
				SE_CORE_ERROR("VULKAN :: VALIDATION :: {0}", pCallbackData->pMessage);
				break;
			case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
				SE_CORE_ERROR("VULKAN :: VALIDATION :: {0}", pCallbackData->pMessage);
				break;
			default:
				break;
			}
			//std::cerr << "VULKAN :: validation layer :: " << pCallbackData->pMessage << std::endl;

			return VK_FALSE;
		}

		VkInstance IGraphicContextVK::instance;

		auto IGraphicContextVK::getVKInstance()->VkInstance&
		{
			return instance;
		}

		auto IGraphicContextVK::hasValidationLayers() -> bool
		{
			return true;
		}

		auto IGraphicContextVK::initialize() -> bool
		{
			createInstance();
			setupDebugMessenger();

			return true;
		}

		void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
			createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			createInfo.pfnUserCallback = debugCallback;
		}

		auto IGraphicContextVK::createInstance() -> void
		{
			if (enableValidationLayers && !checkValidationLayerSupport()) {
				SE_CORE_ERROR("validation layers requested, but not available!");
			}
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
			// specify the desired global extensions
			auto extensions = getRequiredExtensions();
			VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
			
			// not optional
			// Tells the Vulkan driver which global extensions and validation layers we want to use.
			// Global here means that they apply to the entire program and not a specific device,
			VkInstanceCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			createInfo.pApplicationInfo = &appInfo;
			createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
			createInfo.ppEnabledExtensionNames = extensions.data();
			// determine the global validation layers to enable
			if (enableValidationLayers) {
				createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();
				// add debug messenger for init
				populateDebugMessengerCreateInfo(debugCreateInfo);
				createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
			}
			else {
				createInfo.enabledLayerCount = 0;
				createInfo.pNext = nullptr;
			}

			// all prepared
			// issue the vkCreateInstance call
			VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
			// check error
			if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
				SE_CORE_ERROR("failed to create instance!");
			}
		}

		auto IGraphicContextVK::checkExtension() -> void
		{
			// get extension count
			uint32_t extensionCount = 0;
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
			// get extension details
			std::vector<VkExtensionProperties> extensions(extensionCount);
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
			// start listing extensions:
			SE_CORE_INFO("╭─ VULKAN :: Available extensions");
			for (const auto& extension : extensions) {
				SE_CORE_INFO("│\t{0}", extension.extensionName);
			}
			SE_CORE_INFO("╰─ VULKAN :: INFO END");
		}

		auto IGraphicContextVK::checkValidationLayerSupport() -> bool
		{
			// get extension count
			uint32_t layerCount;
			vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
			// get extension details
			std::vector<VkLayerProperties> availableLayers(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
			// for each possible name
			for (const char* layerName : validationLayers) {
				bool layerFound = false;
				// compare with every abailable layer name
				for (const auto& layerProperties : availableLayers) {
					if (strcmp(layerName, layerProperties.layerName) == 0) {
						layerFound = true;
						break;
					}
				}
				// layer not found
				if (!layerFound) {
					return false;
				}
			}
			// find validation layer
			return true;
		}

		auto IGraphicContextVK::getRequiredExtensions()->std::vector<const char*>
		{
			uint32_t glfwExtensionCount = 0;
			const char** glfwExtensions;
			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
			// add extensions that glfw needs
			std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
			// add extensions that validation layer needs
			if (enableValidationLayers) {
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}
			// finialize collection
			return extensions;
		}

		VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
			auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
			if (func != nullptr) {
				return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
			}
			else {
				return VK_ERROR_EXTENSION_NOT_PRESENT;
			}
		}

		void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
			auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
			if (func != nullptr) {
				func(instance, debugMessenger, pAllocator);
			}
		}

		auto IGraphicContextVK::setupDebugMessenger() -> void
		{
			if (!enableValidationLayers) return;
			
			VkDebugUtilsMessengerCreateInfoEXT createInfo;
			populateDebugMessengerCreateInfo(createInfo);

			// load function from extern
			if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
				SE_CORE_ERROR("failed to set up debug messenger!");
			}
		}

		auto IGraphicContextVK::createSurface() -> void
		{
			//if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			//	SE_CORE_ERROR("failed to create window surface!");
			//}
		}

		auto IGraphicContextVK::destroy() -> bool
		{
			if (enableValidationLayers) {
				DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
			}

			vkDestroyInstance(instance, nullptr);
			return true;
		}
	}
}