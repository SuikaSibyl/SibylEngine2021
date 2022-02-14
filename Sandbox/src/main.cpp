module;
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <Macros.h>
#include <EntryPoint.h>
#include <string_view>
module Main;
import Core.Assert;
import Core.Test;
import Core.Window;
import Core.Enums;
import Core.Event;
import Core.SObject;
import Core.SPointer;
import Core.Window;
import Core.Layer;
import Core.LayerStack;
import Core.Application;
import RHI.GraphicContext.VK;
import RHI.GraphicContext.DX12;
import RHI.IPhysicalDevice.DX12;
import RHI.IPhysicalDevice.VK;
import RHI.ILogicalDevice.VK;
import RHI.ILogicalDevice.DX12;
import RHI.ISwapChain.VK;
import RHI.ICompileSession;

using namespace SIByL;

class SandboxApp :public IApplication
{
public:
	virtual void onAwake() override
	{
		WindowLayer* window_layer = new WindowLayer(
			SIByL::EWindowVendor::GLFW,
			onEventCallbackFn,
			1280,
			720,
			"Hello");

		pushLayer(window_layer);
		window_layers.push_back(window_layer);

		auto vk_context = CreateScope<RHI::IGraphicContextVK>();
		vk_context->attachWindow(window_layer->getWindow());
		auto vk_device = CreateScope<RHI::IPhysicalDeviceVK>(vk_context.get());
		auto vk_logical_device = CreateScope<RHI::ILogicalDeviceVK>(vk_device.get());
		auto vk_swapbuffer = CreateScope<RHI::ISwapChainVK>(vk_logical_device.get());

		auto dx12_context = CreateScope<RHI::IGraphicContextDX12>();
		auto dx12_device = CreateScope<RHI::IPhysicalDeviceDX12>(dx12_context.get());
		auto dx12_logical_device = CreateScope<RHI::ILogicalDeviceDX12>(dx12_device.get());

		RHI::SLANG::ICompileSession compile_session;
		compile_session.loadModule("hello-world", "computeMain");
	}

	virtual bool onWindowClose(WindowCloseEvent& e) override
	{
		for (auto iter = window_layers.begin(); iter != window_layers.end(); iter++)
		{
			if ((*iter)->getWindow() == (IWindow*)e.window)
			{
				delete (*iter);
				layer_stack.popLayer((ILayer*)*iter);
				window_layers.erase(iter);
				break;
			}
		}

		if (window_layers.size() == 0)
			is_running = false;
		return true;
	}

private:
	std::vector<WindowLayer*> window_layers;
};

auto SE_CREATE_APP() noexcept -> SIByL::IApplication*
{
#ifdef _DEBUG
	SE_TEST_EXEC_ALL();
#endif

	SandboxApp* app = new SandboxApp();
	return app;
}