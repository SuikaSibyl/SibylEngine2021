module;
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IFramebuffer.VK;
import Core.Color;
import RHI.IResource;
import RHI.IFramebuffer;
import RHI.IRenderPass;
import RHI.IRenderPass.VK;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
import RHI.ITexture;
import RHI.IRenderPass;
import RHI.ITextureView;
import RHI.ITextureView.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IFramebufferVK :public IFramebuffer
		{
		public:
			IFramebufferVK(FramebufferDesc const&, ILogicalDeviceVK*);
			IFramebufferVK(IFramebufferVK&&) = default;
			virtual ~IFramebufferVK();

			auto getVkFramebuffer() noexcept -> VkFramebuffer*;
			auto getSize(uint32_t& width, uint32_t& height) noexcept -> void;
		private:
			auto createFramebuffers() noexcept -> void;
			ILogicalDeviceVK* logicalDevice;
			VkFramebuffer framebuffer;
			FramebufferDesc desc;
			std::vector<VkImageView> attachments;
		};
	}
}
