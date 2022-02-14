module;
#include <dxgi1_6.h>
export module RHI.GraphicContext.DX12;
import RHI.GraphicContext;
import Core.Window;
import Core.Window.WIN;

namespace SIByL
{
	namespace RHI
	{
		export class IGraphicContextDX12 :public IGraphicContext
		{
		public:
			// SObject
			virtual auto initialize() -> bool;
			virtual auto destroy() -> bool;
			// IGraphicContext
			virtual auto attachWindow(IWindow* window) noexcept -> void override;
			// IGraphicContextDX12
			auto getDXGIFactory() noexcept -> IDXGIFactory6*;

		private:
			IDXGIFactory6* pDXGIFactory = nullptr;
			IWindowWIN* windowAttached;
		};
	}
}