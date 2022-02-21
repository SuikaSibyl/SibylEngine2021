module;
#include <Vendors/DX12/WindowsPlatform.h>
#include <vector>
#include <dxgi1_6.h>
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

	IPhysicalDeviceDX12::IPhysicalDeviceDX12(IGraphicContext* graphic_context)
	{
		graphicContext = static_cast<IGraphicContextDX12*>(graphic_context);
	}

	auto IPhysicalDeviceDX12::initialize() -> bool
	{
		// set debugLayer according to compile mode
		enableDebugLayer = enableValidationLayers;

		queryAllAdapters();
		// If the only adapter we found is a software adapter, log error message for QA
		if (!enumedDXGIAdapters.size() && foundSoftwareAdapter)
		{
			SE_CORE_ASSERT(0, "The only available GPU has DXGI_ADAPTER_FLAG_SOFTWARE. Early exiting");
			return false;
		}
		else
		{
			selectedAdapter = enumedDXGIAdapters[0];
		}

		return true;
	}

	auto IPhysicalDeviceDX12::destroy() -> bool
	{
		for (auto iter : enumedDXGIAdapters)
		{
			SAFE_RELEASE(iter);
		}

		return true;
	}

	auto IPhysicalDeviceDX12::queryAllAdapters() noexcept -> void
	{
		IDXGIAdapter4* adapter = NULL;
		// Use DXGI6 interface which lets us specify gpu preference
		for (UINT i = 0; graphicContext->getDXGIFactory()->EnumAdapterByGpuPreference(i,
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
				enumedDXGIAdapters.push_back(adapter);
		}
	}

	auto IPhysicalDeviceDX12::isDebugLayerEnabled() noexcept -> bool
	{
		return enableDebugLayer;
	}
	
	auto IPhysicalDeviceDX12::getGraphicContext() noexcept -> IGraphicContext*
	{
		return (IGraphicContext*)graphicContext;
	}

	auto IPhysicalDeviceDX12::getGraphicContextDX12() noexcept -> IGraphicContextDX12*
	{
		return graphicContext;
	}

	auto IPhysicalDeviceDX12::getAdapter() noexcept -> IDXGIAdapter4*
	{
		return selectedAdapter;
	}
}