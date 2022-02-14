module;
#include <dxgi1_6.h>
#include <Vendors/DX12/WindowsPlatform.h>
module RHI.GraphicContext.DX12;
import RHI.GraphicContext;
import Core.Window;
import Core.Window.WIN;

namespace SIByL::RHI
{
#ifdef _DEBUG
	const bool enableValidationLayers = true;
#else
	const bool enableValidationLayers = false;
#endif

	auto IGraphicContextDX12::initialize() -> bool
	{		
		// set debugLayer according to compile mode
		bool enableDebugLayer = enableValidationLayers;
		UINT flags = 0;
		if (enableDebugLayer) flags = DXGI_CREATE_FACTORY_DEBUG;

		// create dxgiFactory
		if (SUCCEEDED(CreateDXGIFactory2(flags, IID_PPV_ARGS(&pDXGIFactory))))
			return true;
		else return false;
	}

	auto IGraphicContextDX12::destroy() -> bool
	{
		SAFE_RELEASE(pDXGIFactory);
		return true;
	}

	// IGraphicContext
	auto IGraphicContextDX12::attachWindow(IWindow* window) noexcept -> void
	{
		return;
	}

	auto IGraphicContextDX12::getDXGIFactory() noexcept -> IDXGIFactory6*
	{
		return pDXGIFactory;
	}
}