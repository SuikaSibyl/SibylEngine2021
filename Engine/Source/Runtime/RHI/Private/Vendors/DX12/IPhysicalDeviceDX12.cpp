module;
#include "WindowsPlatform.h"
#include <vector>
#include <string>
module RHI.IPhysicalDevice.DX12;
import Core.Log;
import Core.Assert;

namespace SIByL::RHI
{
#ifdef _DEBUG
	const bool enableValidationLayers = true;
#else
	const bool enableValidationLayers = false;
#endif

	auto IPhysicalDeviceDX12::initialize() -> bool
	{
		// set debugLayer according to compile mode
		enableDebugLayer = enableValidationLayers;
		UINT flags = 0;
		if (enableDebugLayer) flags = DXGI_CREATE_FACTORY_DEBUG;

		// create dxgiFactory
		if (SUCCEEDED(CreateDXGIFactory2(flags, IID_PPV_ARGS(&pDXGIFactory))))
		{
			queryAllAdapters();
			// If the only adapter we found is a software adapter, log error message for QA
			if (!DXGIAdapters.size() && foundSoftwareAdapter)
			{
				SE_CORE_ASSERT(0, "The only available GPU has DXGI_ADAPTER_FLAG_SOFTWARE. Early exiting");
				return false;
			}
		}
		return true;
	}

	auto IPhysicalDeviceDX12::destroy() -> bool
	{
		for (auto iter : DXGIAdapters)
		{
			SAFE_RELEASE(iter);
		}
		SAFE_RELEASE(pDXGIFactory);
		return true;
	}

	auto IPhysicalDeviceDX12::queryAllAdapters() noexcept -> void
	{
		IDXGIAdapter4* adapter = NULL;
		// Use DXGI6 interface which lets us specify gpu preference
		for (UINT i = 0; pDXGIFactory->EnumAdapterByGpuPreference(i,
			DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
			IID_PPV_ARGS(&adapter)) != DXGI_ERROR_NOT_FOUND;
			i++)
		{
			DXGI_ADAPTER_DESC3 desc = { 0 };
			adapter->GetDesc3(&desc);
			SE_CORE_INFO(L"DX12 :: Physical Device Found, {0}", std::wstring(desc.Description));
			if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
			{
				foundSoftwareAdapter = true;
				SAFE_RELEASE(adapter);
			}
			else
				DXGIAdapters.push_back(adapter);
		}
	}

	auto IPhysicalDeviceDX12::isDebugLayerEnabled() noexcept -> bool
	{
		return enableDebugLayer;
	}

}