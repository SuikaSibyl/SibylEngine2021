module;
#pragma comment(lib, "dxgi.lib")
#include <dxgi1_6.h>
#include <vector>
export module RHI.IPhysicalDevice.DX12;
import RHI.IPhysicalDevice;

namespace SIByL
{
	namespace RHI
	{
		export class IPhysicalDeviceDX12 :public IPhysicalDevice
		{
		public:
			virtual ~IPhysicalDeviceDX12() = default;

			virtual auto initialize() -> bool;
			virtual auto destroy() -> bool;
			virtual auto isDebugLayerEnabled() noexcept -> bool;

		private:
			bool enableDebugLayer;
			bool foundSoftwareAdapter;
			IDXGIFactory6* pDXGIFactory = nullptr;
			std::vector<IDXGIAdapter4*> DXGIAdapters;

		private:
			auto queryAllAdapters() noexcept -> void;
		};
	}
}