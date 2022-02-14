module;
#include <d3d12.h>
export module RHI.ILogicalDevice.DX12;
import RHI.ILogicalDevice;
import RHI.IPhysicalDevice;
import RHI.IPhysicalDevice.DX12;
import RHI.GraphicContext.DX12;

namespace SIByL
{
	namespace RHI
	{
		export class ILogicalDeviceDX12 :public ILogicalDevice
		{
		public:
			ILogicalDeviceDX12(IPhysicalDevice* physicalDevice);
			virtual ~ILogicalDeviceDX12();

			virtual auto initialize() -> bool;
			virtual auto destroy() -> bool;

			auto getDeviceHandle() noexcept -> ID3D12Device*;
			auto getPhysicalDevice() noexcept -> IPhysicalDeviceDX12*;

		private:
			IGraphicContextDX12* graphicContext;
			IPhysicalDeviceDX12* physicalDevice;
			ID3D12Device* device;

		private:
			auto createLogicalDevice(IPhysicalDeviceDX12* physicalDevice) noexcept -> void;
		};
	}
}