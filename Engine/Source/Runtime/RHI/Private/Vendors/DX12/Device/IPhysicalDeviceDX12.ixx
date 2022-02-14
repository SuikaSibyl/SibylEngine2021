module;
#include <dxgi1_6.h>
#include <vector>
#pragma comment(lib, "D3D12.lib")
export module RHI.IPhysicalDevice.DX12;
import RHI.IPhysicalDevice;
import RHI.GraphicContext;
import RHI.GraphicContext.DX12;

namespace SIByL
{
	namespace RHI
	{
		export class IPhysicalDeviceDX12 :public IPhysicalDevice
		{
		public:
			IPhysicalDeviceDX12(IGraphicContext* graphicContext);
			virtual ~IPhysicalDeviceDX12() = default;

			virtual auto initialize() -> bool;
			virtual auto destroy() -> bool;
			virtual auto isDebugLayerEnabled() noexcept -> bool;
			// IPhysicalDeviceDX12
			auto getGraphicContext() noexcept -> IGraphicContextDX12*;
			auto getAdapter() noexcept -> IDXGIAdapter4*;

		private:
			bool enableDebugLayer;
			bool foundSoftwareAdapter;
			std::vector<IDXGIAdapter4*> enumedDXGIAdapters;
			IDXGIAdapter4* selectedAdapter;
			IGraphicContextDX12* graphicContext;

		private:
			auto queryAllAdapters() noexcept -> void;
		};
	}
}