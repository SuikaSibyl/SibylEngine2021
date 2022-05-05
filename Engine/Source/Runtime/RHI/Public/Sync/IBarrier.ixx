module;
#include <vector>
#include <cstdint>
export module RHI.IBarrier;
import RHI.IEnum;
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
