module;
#include <vulkan/vulkan.h>
export module RHI.ITextureView.VK;
import Core.SObject;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class ITextureViewVK :public ITextureView
		{
		public:
			ITextureViewVK(ILogicalDeviceVK* _logical_device);
			ITextureViewVK(ITexture* texture, ILogicalDeviceVK* _logical_device, ImageUsageFlags extra_usages = 0);
			ITextureViewVK(ITexture* texture, ILogicalDeviceVK* _logical_device, ImageSubresourceRange const& range);
			ITextureViewVK(ITextureViewVK&&) = default;
			ITextureViewVK(ITextureViewVK const&) = delete;
			virtual ~ITextureViewVK();

			auto getpVkImageView() noexcept -> VkImageView*;

		private:
			VkImageView imageView;
			ILogicalDeviceVK* logicalDevice;
		};
	}
}
