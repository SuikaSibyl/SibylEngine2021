export module RHI.GraphicContext;
import Core.SObject;
import Core.Window;

namespace SIByL
{
	namespace RHI
	{
		// The context generally allows you to access the API's inner classes.
		// ╭──────────────┬──────────────────╮
		// │  Vulkan	  │   vk::Instance   │
		// │  DirectX 12  │   IDXGIFactory   │
		// │  OpenGL      │   Varies by OS   │
		// ╰──────────────┴──────────────────╯

		export class IGraphicContext :public SObject
		{
		public:
			virtual ~IGraphicContext() = default;

			virtual auto attachWindow(IWindow* window) noexcept -> void = 0;

		private:

		};
	}
}