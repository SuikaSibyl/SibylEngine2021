module;
#include <cstdint>
export module RHI.IFence;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// ╔═════════════════╗
		// ║      Fence      ║
		// ╚═════════════════╝
		// Fences are objects used to synchronize the CPU and GPU. 
		// Both the CPU and GPU can be instructed to wait at a fence so that the other can catch up. 
		// This can be used to manage resource allocation and deallocation, 
		// making it easier to manage overall graphics memory usage. [Satran et al. 2018]
		// 
		// To signal a fence, all previously submitted commands to the queue must complete.
		// We will also get a full memory barrier that all pending writes are made available
		//
		// However, fence would not make memory available to th CPU
		// Therefore if we do a CPU read, an extra barrier with ACCESS_HOST_READ_BIT flag should be used.
		// In mental model, we can think this as flushing GPU L2 cache out to GPU main memory
		// 
		// ╭──────────────┬──────────────────╮
		// │  Vulkan	  │   vk::Fence      │
		// │  DirectX 12  │   ID3D12Fence    │
		// │  OpenGL      │   glFenceSync    │
		// ╰──────────────┴──────────────────╯

		export class IFence :public SObject
		{
		public:
			IFence() = default;
			IFence(IFence&&) = delete;
			virtual ~IFence() = default;

			virtual auto wait() noexcept -> void = 0;
			virtual auto reset() noexcept -> void = 0;
		};
	}
}
