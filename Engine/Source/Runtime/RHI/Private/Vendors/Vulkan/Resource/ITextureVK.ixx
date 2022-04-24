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
import RHI.IMemoryBarrier;

namespace SIByL
{
	namespace RHI
	{
		export class ITextureVK :public ITexture
		{
		public:
			ITextureVK() = default;
			ITextureVK(Image* image, ILogicalDeviceVK* _logical_device);
			ITextureVK(TextureDesc const&, ILogicalDeviceVK* _logical_device);
			ITextureVK(VkImage _image, IResourceVK&& _resource, TextureDesc const& desc, ILogicalDeviceVK* _logical_device);
			ITextureVK(ITextureVK const&) = delete;
			ITextureVK(ITextureVK &&);
			virtual ~ITextureVK();

			virtual auto getNativeHandle() noexcept -> uint64_t override;
			virtual auto transitionImageLayout(ImageLayout old_layout, ImageLayout new_layout) noexcept -> void override;
			auto getVkImage() noexcept -> VkImage* { return &image; }
			virtual auto getDescription() noexcept -> TextureDesc const& override { return desc; }
			
		private:
			TextureDesc desc;
			ILogicalDeviceVK* logicalDevice;
			VkImage image;
			VkDeviceMemory deviceMemory;
			IResourceVK resource;
			bool externalImage = false;
		};

		export inline auto getVkImageAspectFlags(ImageAspectFlags) noexcept -> VkImageAspectFlags;
		export struct IBufferImageCopyVK :public IBufferImageCopy
		{
		public:
			IBufferImageCopyVK(BufferImageCopyDesc const&);
			auto getVkBufferImageCopy() noexcept -> VkBufferImageCopy* { return &copy; }
		private:
			VkBufferImageCopy copy;
		};
	}
}
