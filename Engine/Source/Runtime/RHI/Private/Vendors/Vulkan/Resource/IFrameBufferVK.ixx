module;
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IFramebuffer.VK;
import RHI.IResource;
import RHI.IFramebuffer;
import RHI.IRenderPass;
import RHI.IRenderPass.VK;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IFramebufferVK :public IFramebuffer
		{
		public:
			IFramebufferVK() = default;
			IFramebufferVK(IFramebufferVK&&) = default;
			virtual ~IFramebufferVK();

			auto getVkFramebuffer() noexcept -> VkFramebuffer*;
			auto getSize(uint32_t& width, uint32_t& height) noexcept -> void;
		private:
			auto createFramebuffers() noexcept -> void;
			IRenderPassVK* renderPass;
			ILogicalDeviceVK* logicalDevice;
			VkFramebuffer framebuffer;
			uint32_t width, height;
			std::vector<VkImageView> attachments;
		};
	}
}
