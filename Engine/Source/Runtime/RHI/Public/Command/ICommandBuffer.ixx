module;
#include <cstdint>
export module RHI.ICommandBuffer;
import RHI.IResource;
import RHI.IRenderPass;
import RHI.IFramebuffer;
import RHI.IPipeline;
import RHI.ISemaphore;
import RHI.IFence;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;
import RHI.IBuffer;
import RHI.IPipelineLayout;
import RHI.IDescriptorSet;

namespace SIByL
{
	namespace RHI
	{
		// A Command Buffer is an asynchronous computing unit, 
		// where you describe procedures for the GPU to execute,
		// such as draw calls, copying data from CPU-GPU accessible memory to GPU exclusive memory,
		// and set various aspects of the graphics pipeline dynamically such as the current scissor.
		// 
		// Previously you would declare what you wanted the GPU to execute procedurally and it would do those tasks, 
		// but GPUs are inherently asynchronous, so the driver would have been responsible for figuring out when to schedule tasks to the GPU.
		//
		// ╭──────────────┬─────────────────────────────────────────────────╮
		// │  Vulkan	  │   vk::CommandBuffer                             │
		// │  DirectX 12  │   ID3D12GraphicsCommandList                     │
		// │  OpenGL      │   Intenal to Driver or with GL_NV_command_list  │
		// ╰──────────────┴─────────────────────────────────────────────────╯

		export class ICommandBuffer :public IResource
		{
		public:
			ICommandBuffer() = default;
			ICommandBuffer(ICommandBuffer&&) = default;
			virtual ~ICommandBuffer() = default;

			virtual auto reset() noexcept -> void = 0;
			virtual auto submit() noexcept -> void = 0;
			virtual auto submit(ISemaphore* wait, ISemaphore* signal, IFence* fence) noexcept -> void = 0;
			virtual auto beginRecording(CommandBufferUsageFlags flags = 0) noexcept -> void = 0;
			virtual auto endRecording() noexcept -> void = 0;
			virtual auto cmdBeginRenderPass(IRenderPass* render_pass, IFramebuffer* framebuffer) noexcept -> void = 0;
			virtual auto cmdEndRenderPass() noexcept -> void = 0;
			virtual auto cmdBindPipeline(IPipeline* pipeline) noexcept -> void = 0;
			virtual auto cmdBindVertexBuffer(IVertexBuffer* buffer) noexcept -> void = 0;
			virtual auto cmdBindIndexBuffer(IIndexBuffer* buffer) noexcept -> void = 0;
			virtual auto cmdDraw(uint32_t const& vertex_count, uint32_t const& instance_count,
				uint32_t const& first_vertex, uint32_t const& first_instance) noexcept -> void = 0;
			virtual auto cmdDrawIndexed(uint32_t const& index_count, uint32_t const& instance_count,
				uint32_t const& first_index, uint32_t const& index_offset, uint32_t const& first_instance) noexcept -> void = 0;
			virtual auto cmdCopyBuffer(IBuffer* src, IBuffer* dst, uint32_t const& src_offset, uint32_t const& dst_offset, uint32_t const& size) noexcept -> void = 0;
			virtual auto cmdBindDescriptorSets(PipelineBintPoint point, IPipelineLayout* pipeline_layout, uint32_t const& idx_first_descset, uint32_t const& count_sets_to_bind, IDescriptorSet** sets, uint32_t const&, uint32_t const*) noexcept -> void = 0;
		};
	}
}
