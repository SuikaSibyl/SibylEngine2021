module;
#include <vulkan/vulkan.h>
module RHI.ITextureView.VK;
import Core.SObject;
import RHI.ITextureView;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		ITextureViewVK::ITextureViewVK(ILogicalDeviceVK* _logical_device)
			:logicalDevice(_logical_device)
		{}

		ITextureViewVK::~ITextureViewVK()
		{
			if (imageView)
			{
				vkDestroyImageView(logicalDevice->getDeviceHandle(), imageView, nullptr);
			}
		}

		auto ITextureViewVK::getpVkImageView() noexcept -> VkImageView*
		{
			return &imageView;
		}
	}
}
