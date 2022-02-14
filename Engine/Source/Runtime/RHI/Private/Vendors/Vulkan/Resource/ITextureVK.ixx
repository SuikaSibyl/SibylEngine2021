module;
#include <vulkan/vulkan.h>
export module RHI.ITexture.VK;
import RHI.ITexture;
import RHI.IResource.VK;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class ITextureVK :public ITexture
		{
		public:
			ITextureVK() = default;
			ITextureVK(VkImage _image, IResourceVK&& _resource, ILogicalDeviceVK* _logical_device);
			ITextureVK(ITextureVK const&) = delete;
			ITextureVK(ITextureVK &&);
			virtual ~ITextureVK();

		private:
			ILogicalDeviceVK* logicalDevice;
			VkImage image;
			VkImageView imageView;
			IResourceVK resource;
			bool externalImage = false;
		};
	}
}
