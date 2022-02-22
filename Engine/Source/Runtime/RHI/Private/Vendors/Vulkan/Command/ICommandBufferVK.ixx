module;
#include <cstdint>
#include <vulkan/vulkan.h>
export module RHI.ICommandBuffer.VK;
import RHI.ICommandBuffer;
import RHI.ICommandPool;
import RHI.ICommandPool.VK;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
import RHI.IEnum;
import RHI.IRenderPass;
import RHI.IRenderPass.VK;
import RHI.IFramebuffer;
import RHI.IFramebuffer.VK;
import RHI.IPipeline;
import RHI.IPipeline.VK;

namespace SIByL
{
	namespace RHI
	{
		export class ICommandBufferVK :public ICommandBuffer
		{
		public:
			ICommandBufferVK(ICommandPoolVK*, ILogicalDeviceVK*);
			ICommandBufferVK(ICommandBufferVK const&) = delete;
			ICommandBufferVK(ICommandBufferVK&&) = delete;
			virtual ~ICommandBufferVK() = default;
			// ICommandBuffer
			virtual auto beginRecording() noexcept -> void override;
			virtual auto endRecording() noexcept -> void override;
			virtual auto cmdBeginRenderPass(IRenderPass* render_pass, IFramebuffer* framebuffer) noexcept -> void override;
			virtual auto cmdEndRenderPass() noexcept -> void override;
			virtual auto cmdBindPipeline(IPipeline* pipeline) noexcept -> void override;
			virtual auto cmdDraw(uint32_t const& vertex_count, uint32_t const& instance_count,
				uint32_t const& first_vertex, uint32_t const& first_instance) noexcept -> void override;

		private:
			auto createVkCommandBuffer() noexcept -> void;
			ILogicalDeviceVK* logicalDevice;
			ICommandPoolVK* commandPool;
			VkCommandBuffer commandBuffer;
		};
	}
}
