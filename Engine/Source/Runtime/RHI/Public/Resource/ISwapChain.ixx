module;
#include <cstdint>
export module RHI.ISwapChain;
import RHI.IResource;
import RHI.IEnum;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.ISemaphore;

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

		export struct SwapchainDesc
		{
			unsigned int width = 1280;
			unsigned int height = 720;
		};

		export class ISwapChain :public IResource
		{
		public:
			virtual ~ISwapChain() = default;
			virtual auto getExtend() noexcept -> Extend = 0;
			virtual auto getSwapchainCount() noexcept -> unsigned int = 0;
			virtual auto getITexture(unsigned int idx) noexcept ->ITexture* = 0;
			virtual auto getITextureView(unsigned int idx) noexcept ->ITextureView* = 0;

			virtual auto acquireNextImage(ISemaphore* semaphore) noexcept -> uint32_t = 0;
			virtual auto present(uint32_t const& idx, ISemaphore* semaphore) noexcept -> void = 0;
		};
	}
}
