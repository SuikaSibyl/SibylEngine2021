export module RHI.GraphicContext;
import Core.SObject;
import Core.Window;
import RHI.IEnum;

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
			auto setAPI(API const& _api) noexcept -> void { api = _api; }
			auto getAPI() noexcept -> API { return api; }

		private:
			API api;
		};
	}
}