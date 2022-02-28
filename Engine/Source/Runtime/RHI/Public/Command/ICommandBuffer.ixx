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
import RHI.IBarrier;
import RHI.ITexture;

namespace SIByL
{
	namespace RHI
	{
		// ╔══════════════════════════╗
		// ║      Command Buffer      ║
		// ╚══════════════════════════╝
		// A Command Buffer is a object used to record commands.
		// 
		// ╭──────────────┬─────────────────────────────────────────────────╮
		// │  Vulkan	  │   vk::CommandBuffer                             │
		// │  DirectX 12  │   ID3D12GraphicsCommandList                     │
		// │  OpenGL      │   Intenal to Driver or with GL_NV_command_list  │
		// ╰──────────────┴─────────────────────────────────────────────────╯
		// 
		//  ╭╱───────────────╲╮
		//  ╳    Hierarchy    ╳
		//  ╰╲───────────────╱╯
		// For Vulkan, there are two levels, primary & secondary
		// Secondary command buffers can be executed by primary command buffers.
		// Therefore, they are not directly submitted to queues
		// 
		//  ╭╱───────────────╲╮
		//  ╳    Commands     ╳
		//  ╰╲───────────────╱╯
		// Recorded commands include commands to :
		//  ► bind pipelines & descriptor sets to the command buffer
		//  ► modify dynamic state
		//  ► draw
		//  ► dispatch
		//  ► execute secondary command buffers
		//  ► copy buffers and images
		// 
		//  ╭╱───────────────╲╮
		//  ╳    Lifecycle    ╳
		//  ╰╲───────────────╱╯
		// ► Initial: after allocated, can only be moved to the recording state or freed
		// ► Recording: from initial state, cmd* commands can be used to record
		// ► Executable: from recording state, can be submitted, reset, or recorded to another command buffer
		// ► Pending: after submission, applications must not modify the buffer in any way, after exectued, to executable / invalid states
		// ► Invalid: can only be reset or freed
		// 
		// ╭─────────────────╮
		// │  Command Queue  │
		// ╰─────────────────╯
		// Command buffers can be subsequently submitted to a device queue for execution
		// 
		// where you describe procedures for the GPU to execute,
		// such as draw calls, copying data from CPU-GPU accessible memory to GPU exclusive memory,
		// and set various aspects of the graphics pipeline dynamically such as the current scissor.
		// 
		// Previously you would declare what you wanted the GPU to execute procedurally and it would do those tasks, 
		// but GPUs are inherently asynchronous, so the driver would have been responsible for figuring out when to schedule tasks to the GPU.
		//
		//  ╭╱───────────────────────────────╲╮
		//  ╳    Implicit Memory Guarantee    ╳
		//  ╰╲───────────────────────────────╱╯
		// Submitting commands to a queue makes all memory acess 
		// performed by host visible to all stages and access masks.
		// 
		// Basically submitting a batch issues a cache invalidation on 
		// host visible memory.
		//
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
			virtual auto cmdPipelineBarrier(IBarrier* barrier) noexcept -> void = 0;
			virtual auto cmdCopyBufferToImage(IBuffer* buffer, ITexture* image, IBufferImageCopy*) noexcept -> void = 0;
		};
	}
}
