module;

export module RHI.ISemaphore;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// ╔═════════════════════╗
		// ║      Semaphore      ║
		// ╚═════════════════════╝
		// Semaphores are objects used introduce dependencies between operations, 
		// it actually facilitate GPU <-> GPU synchronization,
		// such as waiting before acquiring the next image in the swapchain 
		// before submitting command buffers to your device queue.
		// 
		// To signal a semaphore, all previously submitted commands to the queue must complete.
		// We will also get a full memory barrier that all pending writes are made available
		// 
		// While signaling a semaphore makes all memory abailable
		// waiting for a semaphore makes memory visible.
		// Therefore, no extra barrier is need if we use a semaphore.
		// 
		// Vulkan is unique in that semaphores are a part of the API, 
		// with DirectX and Metal delegating that to OS calls.
		//
		// ╭──────────────┬───────────────────╮
		// │  Vulkan	  │   vk::Semaphore   │
		// │  DirectX 12  │   HANDLE          │
		// │  OpenGL      │   Varies by OS    │
		// ╰──────────────┴───────────────────╯

		export class ISemaphore :public SObject
		{
		public:
			ISemaphore() = default;
			ISemaphore(ISemaphore&&) = default;
			virtual ~ISemaphore() = default;

		};
	}
}
