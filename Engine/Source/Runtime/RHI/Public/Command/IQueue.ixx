module;

export module RHI.IQueue;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// A Queue allows you to enqueue tasks for the GPU to execute.
		// ╭──────────────┬────────────────────────╮
		// │  Vulkan	  │   vk::Queue            │
		// │  DirectX 12  │   ID3D12CommandQueue   │
		// │  OpenGL      │   N/A                  │
		// ╰──────────────┴────────────────────────╯

		export class IQueue :public SObject
		{
		public:
			IQueue() = default;
			IQueue(IQueue const&) = delete;
			IQueue(IQueue&&) = delete;
			virtual ~IQueue() = default;

		};
	}
}
