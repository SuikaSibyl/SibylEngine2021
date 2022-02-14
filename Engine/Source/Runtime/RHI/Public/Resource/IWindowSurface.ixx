module;

export module RHI.IWindowSurface;
import RHI.IResource;

namespace SIByL
{
	namespace RHI
	{
		// A window Surface allows you to bind all draw calls to an OS specific window.
		// ╭──────────────┬───────────────────╮
		// │  Vulkan	  │   vk::Surface     │
		// │  DirectX 12  │   ID3D12Resource  │
		// │  OpenGL      │   Varies by OS    │
		// ╰──────────────┴───────────────────╯

		export class IWindowSurface :public IResource
		{
		public:
			IWindowSurface() = default;
			IWindowSurface(IWindowSurface&&) = default;
			virtual ~IWindowSurface() = default;


		};
	}
}
