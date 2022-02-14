module;
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
module RHI.ICommandBuffer.VK;
import Core.Log;
import RHI.ICommandBuffer;
import RHI.ICommandPool;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
import RHI.IEnum;
import RHI.IRenderPass;
import RHI.IRenderPass.VK;
import RHI.IFramebuffer;
import RHI.IFramebuffer.VK;
import RHI.IPipeline;
import RHI.IPipeline.VK;

namespace SIByL::RHI
{
	auto ICommandBufferVK::beginRecording() noexcept -> void
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		// VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be rerecorded right after executing it once.
		// VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : This is a secondary command buffer that will be entirely within a single render pass.
		// VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : The command buffer can be resubmitted while it is also already pending execution.
		beginInfo.flags = 0; // Optional
		beginInfo.pInheritanceInfo = nullptr; // Optional

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to begin recording command buffer!");
		}
	}

	auto ICommandBufferVK::endRecording() noexcept -> void
	{
		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to record command buffer!");
		}
	}

	auto ICommandBufferVK::cmdBeginRenderPass(IRenderPass* render_pass, IFramebuffer* framebuffer) noexcept -> void
	{
		uint32_t width, height;
		static_cast<IFramebufferVK*>(framebuffer)->getSize(width, height);

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = *static_cast<IRenderPassVK*>(render_pass)->getRenderPass();
		renderPassInfo.framebuffer = *static_cast<IFramebufferVK*>(framebuffer)->getVkFramebuffer();
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = VkExtent2D{ width, height };

		VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
	}

	auto ICommandBufferVK::cmdEndRenderPass() noexcept -> void
	{
		vkCmdEndRenderPass(commandBuffer);
	}
	
	auto ICommandBufferVK::cmdBindPipeline(IPipeline* pipeline) noexcept -> void
	{
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, 
			*static_cast<IPipelineVK*>(pipeline)->getVkPipeline());
	}

	auto ICommandBufferVK::cmdDraw(uint32_t const& vertex_count, uint32_t const& instance_count,
		uint32_t const& first_vertex, uint32_t const& first_instance) noexcept -> void
	{
		vkCmdDraw(commandBuffer, vertex_count, instance_count, first_vertex, first_instance);
	}

	auto ICommandBufferVK::createVkCommandBuffer() noexcept -> void
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = *commandPool->getVkCommandPool();
		// VK_COMMAND_BUFFER_LEVEL_PRIMARY: 
		// - Can be submitted to a queue for execution, but cannot be called from other command buffers.
		// VK_COMMAND_BUFFER_LEVEL_SECONDARY : 
		// - Cannot be submitted directly, but can be called from primary command buffers.
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;

		if (vkAllocateCommandBuffers(logicalDevice->getDeviceHandle(), &allocInfo, &commandBuffer) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to allocate command buffers!");
		}
	}
}
