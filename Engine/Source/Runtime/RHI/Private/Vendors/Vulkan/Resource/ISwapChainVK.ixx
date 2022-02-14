module;
#include <vulkan/vulkan.h>
#include <vector>
export module RHI.ISwapChain.VK;
import Core.Window.GLFW;
import RHI.ISwapChain;
import RHI.IPhysicalDevice.VK;
import RHI.ILogicalDevice.VK;
import RHI.ITexture.VK;

namespace SIByL
{
	namespace RHI
	{
		export class ISwapChainVK :public ISwapChain
		{
		public:
			ISwapChainVK(ILogicalDeviceVK* logical_device);
			virtual ~ISwapChainVK();

		private:
			VkSwapchainKHR swapChain;
			IPhysicalDeviceVK* physicalDevice;
			ILogicalDeviceVK* logicalDevice;
			IWindowGLFW* windowAttached;
			std::vector<ITextureVK> swapChainTextures;
			VkFormat swapChainImageFormat;
			VkExtent2D swapChainExtent;

		private:
			auto createSwapChain() noexcept -> void;
			auto chooseSwapSurfaceFormat(std::vector<VkSurfaceFormatKHR> const& availableFormats) noexcept -> VkSurfaceFormatKHR;
			auto chooseSwapPresentMode(std::vector<VkPresentModeKHR> const& availablePresentModes) noexcept -> VkPresentModeKHR;
			auto chooseSwapExtent(VkSurfaceCapabilitiesKHR const& capabilities) noexcept -> VkExtent2D;
		};
	}
}
