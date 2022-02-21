module;
#include <vulkan/vulkan.h>
#include <vector>
export module RHI.ISwapChain.VK;
import Core.Window.GLFW;
import Core.MemoryManager;
import RHI.ISwapChain;
import RHI.IPhysicalDevice.VK;
import RHI.ILogicalDevice.VK;
import RHI.ITexture;
import RHI.ITexture.VK;
import RHI.ITextureView;
import RHI.ITextureView.VK;
import Core.SPointer;

namespace SIByL
{
	namespace RHI
	{
		export class ISwapChainVK :public ISwapChain
		{
		public:
			ISwapChainVK(ILogicalDeviceVK* logical_device);
			virtual ~ISwapChainVK();

			virtual auto getExtend() noexcept -> Extend override;

		private:
			VkSwapchainKHR swapChain;
			IPhysicalDeviceVK* physicalDevice;
			ILogicalDeviceVK* logicalDevice;
			IWindowGLFW* windowAttached;
			VkFormat swapChainImageFormat;
			VkExtent2D swapChainExtent;
			std::vector<ITextureVK> swapChainTextures;
			std::vector<MemScope<ITextureView>> swapchainViews;

		private:
			auto createSwapChain() noexcept -> void;
			auto chooseSwapSurfaceFormat(std::vector<VkSurfaceFormatKHR> const& availableFormats) noexcept -> VkSurfaceFormatKHR;
			auto chooseSwapPresentMode(std::vector<VkPresentModeKHR> const& availablePresentModes) noexcept -> VkPresentModeKHR;
			auto chooseSwapExtent(VkSurfaceCapabilitiesKHR const& capabilities) noexcept -> VkExtent2D;
		};
	}
}
