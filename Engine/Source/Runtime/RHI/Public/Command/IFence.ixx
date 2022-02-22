module;
#include <cstdint>
export module RHI.IFence;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		// Fences are objects used to synchronize the CPU and GPU. 
		// Both the CPU and GPU can be instructed to wait at a fence so that the other can catch up. 
		// This can be used to manage resource allocation and deallocation, 
		// making it easier to manage overall graphics memory usage. [Satran et al. 2018]
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
