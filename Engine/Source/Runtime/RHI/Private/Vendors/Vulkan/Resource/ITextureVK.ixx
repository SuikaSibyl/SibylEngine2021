module;
#include <vulkan/vulkan.h>
export module RHI.ITexture.VK;
import Core.MemoryManager;
import Core.Image;
import RHI.ITexture;
import RHI.IResource.VK;
import RHI.ILogicalDevice.VK;
import RHI.ITextureView;
import RHI.ITextureView.VK;

namespace SIByL
{
	namespace RHI
	{
		export class ITextureVK :public ITexture
		{
		public:
			ITextureVK() = default;
			ITextureVK(Image* image, ILogicalDeviceVK* _logical_device);
			ITextureVK(VkImage _image, IResourceVK&& _resource, ILogicalDeviceVK* _logical_device);
			ITextureVK(ITextureVK const&) = delete;
			ITextureVK(ITextureVK &&);
			virtual ~ITextureVK();

			virtual auto createView(TextureViewDesc const& desc) noexcept -> MemScope<ITextureView> override;

		private:
			ILogicalDeviceVK* logicalDevice;
			VkImage image;
			VkDeviceMemory deviceMemory;
			IResourceVK resource;
			bool externalImage = false;
		};
	}
}
