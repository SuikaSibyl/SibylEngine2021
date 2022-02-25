module;

export module RHI.ISemaphore;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// Semaphores are objects used introduce dependencies between operations, 
		// such as waiting before acquiring the next image in the swapchain 
		// before submitting command buffers to your device queue.
		// 
		// Vulkan is unique in that semaphores are a part of the API, with DirectX and Metal delegating that to OS calls.
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
