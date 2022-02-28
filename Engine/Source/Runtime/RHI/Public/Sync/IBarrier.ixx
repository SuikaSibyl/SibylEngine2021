module;
#include <vector>
#include <cstdint>
export module RHI.IBarrier;
import RHI.IMemoryBarrier;

namespace SIByL
{
	namespace RHI
	{
		// ╔════════════════════╗
		// ║      Barriers      ║
		// ╚════════════════════╝
		// Barrier is a more granular form of synchronization, inside command buffers.
		// It is not a "resource" but a "command", 
		// it divide all commands on queue into two parts: "before" and "after", 
		// and indicate dependency of {a subset of "after"} on {another subset of "before"}
		// 
		// As a command, input parameters are:
		//  ► PipelineStageFlags    - srcStageMask
		//  ► PipelineStageFlags    - dstStageMask
		//  ► DependencyFlags       - dependencyFlags
		//  ► [MemoryBarrier]       - pBufferMemoryBarriers
		//  ► [BufferMemoryBarrier] - pBufferMemoryBarriers
		//  ► [ImageMemoryBarrier]  - pImageMemoryBarriers
		// 
		// In cmdPipelineBarrier, we are specifying 4 things to happen in order:
		// 1. Wait for srcStageMask to complete
		// 2. Make all writes performed in possible combinations of srcStageMasks + srcAccessMask available
		// 3. Make available memory visible to possible combination of dstStageMask + dstAccessMask
		// 4. Unblock work in dstStageMask
		// 
		// ╭──────────────┬───────────────────────────╮
		// │  Vulkan	  │   vkCmdPipelineBarrier    │
		// │  DirectX 12  │   D3D12_RESOURCE_BARRIER  │
		// │  OpenGL      │   glMemoryBarrier         │
		// ╰──────────────┴───────────────────────────╯
		// 
		//  ╭╱─────────────────────────────────────────╲╮
		//  ╳   Execution Barrier - Source Stage Mask   ╳
		//  ╰╲─────────────────────────────────────────╱╯
		// This present what we are waiting for.
		// Essentially, it is every commands before this one.
		// The mask can restrict the scrope of what we are waiting for.
		//
		//  ╭╱─────────────────────────────────────────╲╮
		//  ╳    Execution Barrier - Dst Stage Mask     ╳
		//  ╰╲─────────────────────────────────────────╱╯
		// Any work submitted after this barrier will need to wait for 
		// the work represented by srcStageMask before it can execute.
		// For example, FRAGMENT_SHADER_BIT, then vertex shading could be executed ealier.
		//
		//  ╭╱────────────────────╲╮
		//  ╳    Memory Barrier    ╳
		//  ╰╲────────────────────╱╯
		// Execution order and memory order are two different things
		// because of multiple & incoherent caches, 
		// synchronizing execution alone is not enough to ensure the different units
		// on GPU can transfer data between themselves.
		//
		// GPU memory write is fistly "available", and only "visible" after cache flushing
		// That is where we should use a MemoryBarrier
		struct BarrierDesc;
		export class IBarrier
		{
		public:
			IBarrier() = default;
			virtual ~IBarrier() = default;

		private:
		};

		// ╔═══════════════════════════╗
		// ║      Pipeline Stages      ║
		// ╚═══════════════════════════╝
		// Pipeline Stage, is a sub-stage of a command.
		// They are used in the barrier command,
		// In a barrier it wait the "before" parts to complete and then execute the "after" part
		// However, using pipeline stages, we only need to wait certain stages of the "before" part 
		// 
		//  ╭╱───────────────────╲╮
		//  ╳    Common Stages    ╳
		//  ╰╲───────────────────╱╯
		// ┌───────────────────────────────┐ ┌─────────────────────────────┐
		// │      COMPUTE / TRANSFER       │ │      RENDER - Fragment      │
		// ├───────────────────────────────┤ ├─────────────────────────────┤
		// │  TOP_OF_PIPE				   │ │  EARLY_FRAGMENT_TESTS       │
		// │  DRAW_INDIRECT				   │ │  FRAGMENT_SHADER		       │
		// │  COMPUTE / TRANSFER           │ │  LATE_FRAGMENT_TESTS        │
		// │  BOTTOM_OF_PIPE		       │ │  COLOR_ATTACHMENT_OUTPUT    │
		// └───────────────────────────────┘ └─────────────────────────────┘
		// ┌───────────────────────────────────────────────────────────────┐
		// │                    RENDER - Geometry		     	           │
		// ├───────────────────────────────────────────────────────────────┤
		// │  DRAW_INDIRECT - Parses indirect buffers				       │
		// │  VERTEX_INPUT - Consumes fixed function VBOs and IBOs	       │
		// │  VERTEX_SHADER - Actual vertex shader					       │
		// │  TESSELLATION_CONTROL_SHADER							       │
		// │  TESSELLATION_EVALUATION_SHADER						       │
		// │  GEOMETRY_SHADER										       │
		// └───────────────────────────────────────────────────────────────┘

		export enum class PipelineStageFlagBits :uint32_t
		{
			TOP_OF_PIPE_BIT = 0x00000001,
			DRAW_INDIRECT_BIT = 0x00000002,
			VERTEX_INPUT_BIT = 0x00000004,
			VERTEX_SHADER_BIT = 0x00000008,
			TESSELLATION_CONTROL_SHADER_BIT = 0x00000010,
			TESSELLATION_EVALUATION_SHADER_BIT = 0x00000020,
			GEOMETRY_SHADER_BIT = 0x00000040,
			FRAGMENT_SHADER_BIT = 0x00000080,
			EARLY_FRAGMENT_TESTS_BIT = 0x00000100,
			LATE_FRAGMENT_TESTS_BIT = 0x00000200,
			COLOR_ATTACHMENT_OUTPUT_BIT = 0x00000400,
			COMPUTE_SHADER_BIT = 0x00000800,
			TRANSFER_BIT = 0x00001000,
			BOTTOM_OF_PIPE_BIT = 0x00002000,
			HOST_BIT = 0x00004000,
			ALL_GRAPHICS_BIT = 0x00008000,
			ALL_COMMANDS_BIT = 0x00010000,
			TRANSFORM_FEEDBACK_BIT_EXT = 0x01000000,
			CONDITIONAL_RENDERING_BIT_EXT = 0x00040000,
			ACCELERATION_STRUCTURE_BUILD_BIT_KHR = 0x02000000,
			RAY_TRACING_SHADER_BIT_KHR = 0x00200000,
			TASK_SHADER_BIT_NV = 0x00080000,
			MESH_SHADER_BIT_NV = 0x00100000,
			FRAGMENT_DENSITY_PROCESS_BIT = 0x00800000,
			FRAGMENT_SHADING_RATE_ATTACHMENT_BIT = 0x00400000,
			COMMAND_PREPROCESS_BIT = 0x00020000,
		};
		export using PipelineStageFlags = uint32_t;

		// ╔════════════════════════════╗
		// ║      Dependency Flags      ║
		// ╚════════════════════════════╝
		// Basically, we could use the NONE flag.

		export enum class DependencyTypeFlagBits:uint32_t
		{
			NONE = 0x00000000,
			BY_REGION_BIT = 0x00000001,
			VIEW_LOCAL_BIT = 0x00000002,
			DEVICE_GROUP_BIT = 0x00000004,
		};
		export using DependencyTypeFlags = uint32_t;

		// ╔════════════════════════╗
		// ║      Barrier Desc      ║
		// ╚════════════════════════╝
		// The Desc will be used to initialize a barrier
		using namespace std;
		export struct BarrierDesc
		{
			// Necessary (Execution Barrier)
			PipelineStageFlags	srcStageMask;
			PipelineStageFlags	dstStageMask;
			DependencyTypeFlags dependencyType;
			// Optional (Memory Barriers)
			vector<IMemoryBarrier*> memoryBarriers;
			vector<IBufferMemoryBarrier*> bufferMemoryBarriers;
			vector<IImageMemoryBarrier*> imageMemoryBarriers;
		};
	}
}
