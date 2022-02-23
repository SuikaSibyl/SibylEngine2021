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
import RHI.IEnum.VK;
import RHI.IRenderPass;
import RHI.IRenderPass.VK;
import RHI.IFramebuffer;
import RHI.IFramebuffer.VK;
import RHI.IPipeline;
import RHI.IPipeline.VK;
import RHI.ISemaphore;
import RHI.ISemaphore.VK;
import RHI.IFence;
import RHI.IFence.VK;
import RHI.IVertexBuffer;
import RHI.IVertexBuffer.VK;
import RHI.IIndexBuffer;
import RHI.IIndexBuffer.VK;
import RHI.IBuffer;
import RHI.IBuffer.VK;

namespace SIByL::RHI
{
	ICommandBufferVK::ICommandBufferVK(ICommandPoolVK* command_pool, ILogicalDeviceVK* logical_device)
		: commandPool(command_pool)
		, logicalDevice(logical_device)
	{
		createVkCommandBuffer();
	}

	auto ICommandBufferVK::reset() noexcept -> void
	{
		vkResetCommandBuffer(commandBuffer, 0);
	}
	
	auto ICommandBufferVK::submit() noexcept -> void
	{
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		
		vkQueueSubmit(*(logicalDevice->getVkGraphicQueue()), 1, &submitInfo, VK_NULL_HANDLE);
	}

	auto ICommandBufferVK::submit(ISemaphore* wait, ISemaphore* signal, IFence* fence) noexcept -> void
	{
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore waitSemaphores[] = { *((ISemaphoreVK*)wait)->getVkSemaphore() };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = waitSemaphores;
		submitInfo.pWaitDstStageMask = waitStages;
		
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		
		VkSemaphore signalSemaphores[] = { *((ISemaphoreVK*)signal)->getVkSemaphore() };
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = signalSemaphores;

		if (vkQueueSubmit(*(logicalDevice->getVkGraphicQueue()), 1, &submitInfo, *((IFenceVK*)fence)->getVkFence()) != VK_SUCCESS) {
			SE_CORE_ERROR("failed to submit draw command buffer!");
		}
	}

	auto ICommandBufferVK::beginRecording(CommandBufferUsageFlags flags) noexcept -> void
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		// VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: The command buffer will be rerecorded right after executing it once.
		// VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT : This is a secondary command buffer that will be entirely within a single render pass.
		// VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : The command buffer can be resubmitted while it is also already pending execution.
		beginInfo.flags = getVkCommandBufferUsageFlags(flags); // Optional
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

	auto ICommandBufferVK::cmdBindVertexBuffer(IVertexBuffer* buffer) noexcept -> void
	{
		VkBuffer vertexBuffers[] = { *((IVertexBufferVK*)buffer)->getVkBuffer()};
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
	}

	auto ICommandBufferVK::cmdBindIndexBuffer(IIndexBuffer* buffer) noexcept -> void
	{
		vkCmdBindIndexBuffer(commandBuffer, *((IIndexBufferVK*)buffer)->getVkBuffer(), 0, ((IIndexBufferVK*)buffer)->getVkIndexType());
	}

	auto ICommandBufferVK::cmdDraw(uint32_t const& vertex_count, uint32_t const& instance_count,
		uint32_t const& first_vertex, uint32_t const& first_instance) noexcept -> void
	{
		vkCmdDraw(commandBuffer, vertex_count, instance_count, first_vertex, first_instance);
	}
	
	auto ICommandBufferVK::cmdDrawIndexed(uint32_t const& index_count, uint32_t const& instance_count,
		uint32_t const& first_index, uint32_t const& index_offset, uint32_t const& first_instance) noexcept -> void
	{
		vkCmdDrawIndexed(commandBuffer, index_count, instance_count, first_index, index_offset, first_instance);
	}

	auto ICommandBufferVK::cmdCopyBuffer(IBuffer* src, IBuffer* dst, uint32_t const& src_offset, uint32_t const& dst_offset, uint32_t const& size) noexcept -> void
	{
		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = src_offset; // Optional
		copyRegion.dstOffset = dst_offset; // Optional
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, *((IBufferVK*)src)->getVkBuffer(), *((IBufferVK*)dst)->getVkBuffer(), 1, &copyRegion);
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
