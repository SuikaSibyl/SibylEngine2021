module;

export module RHI.ICommandQueue;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// ╔═════════════════════════╗
		// ║      Command Queue      ║
		// ╚═════════════════════════╝
		// Command Queue is an abstraction where command buffers are submitted
		// and the GPU churns through commands.
		// It allows you to enqueue tasks for the GPU to execute.
		// 
		// ╭──────────────┬────────────────────────╮
		// │  Vulkan	  │   vkQueue              │
		// │  DirectX 12  │   ID3D12CommandQueue   │
		// │  OpenGL      │   N/A                  │
		// ╰──────────────┴────────────────────────╯
		// 
		//  ╭╱───────────────╲╮
		//  ╳   Synchronize   ╳
		//  ╰╲───────────────╱╯
		// Everything submitted to a queue is a linear stream of commands
		// Any synchronization applies globally to a queue, 
		// and command buffer boundaries are not special in synchronization.
		// Defaultly, all commands in a queue execute out of order, at least in Vulkan.

		export class ICommandQueue :public SObject
		{
		public:
			ICommandQueue() = default;
			virtual ~ICommandQueue() = default;

		};
	}
}
