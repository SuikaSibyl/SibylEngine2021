module;
#include <d3d12.h>
#include <dxgi1_6.h>
#include <vector>
module RHI.ILogicalDevice.DX12;

import RHI.ILogicalDevice;
import RHI.IPhysicalDevice;
import RHI.IPhysicalDevice.DX12;
import RHI.GraphicContext.DX12;

namespace SIByL::RHI
{
	ILogicalDeviceDX12::ILogicalDeviceDX12(IPhysicalDevice* physical_device)
	{
		physicalDevice = dynamic_cast<IPhysicalDeviceDX12*>(physical_device);
		graphicContext = physicalDevice->getGraphicContextDX12();
	}

	ILogicalDeviceDX12::~ILogicalDeviceDX12()
	{

	}

	auto ILogicalDeviceDX12::initialize() -> bool
	{
		createLogicalDevice(physicalDevice);
		return true;
	}

	auto ILogicalDeviceDX12::destroy() -> bool
	{
		return true;
	}

	auto ILogicalDeviceDX12::getDeviceHandle() noexcept -> ID3D12Device*
	{
		return device;
	}

	auto ILogicalDeviceDX12::getPhysicalDeviceDX12() noexcept -> IPhysicalDeviceDX12*
	{
		return physicalDevice;
	}

	auto ILogicalDeviceDX12::getPhysicalDevice() noexcept -> IPhysicalDevice*
	{
		return (IPhysicalDevice*)physicalDevice;
	}

	auto ILogicalDeviceDX12::createLogicalDevice(IPhysicalDeviceDX12* physicalDevice) noexcept -> void
	{
		// create device
		HRESULT hardwareResult = D3D12CreateDevice(
			physicalDevice->getAdapter(),
			D3D_FEATURE_LEVEL_11_0,
			IID_PPV_ARGS(&device)
		);
	}

}