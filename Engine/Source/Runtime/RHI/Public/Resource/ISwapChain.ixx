module;

export module RHI.ISwapChain;
import RHI.IResource;

namespace SIByL
{
	namespace RHI
	{
		// A Swapchain flips between different back buffers for a given window,
		// and controls aspects of rendering such as refresh rate and back buffer swapping behavior.
		// ╭──────────────┬───────────────────╮
		// │  Vulkan	  │   vk::Surface     │
		// │  DirectX 12  │   ID3D12Resource  │
		// │  OpenGL      │   Varies by OS    │
		// ╰──────────────┴───────────────────╯

		export class ISwapChain :public IResource
		{
		public:
			virtual ~ISwapChain() = default;
			
		};
	}
}
