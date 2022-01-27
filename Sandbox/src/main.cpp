#include <iostream>
#include <memory>
#include <EntryPoint.h>

import Core.Application;
import Core.Assert;
import Core.Test;
import Core.Window;
import Core.Enums;
import Core.SObject;
import Core.SPointer;
import RHI.GraphicContext.VK;
import RHI.IPhysicalDevice.DX12;
import RHI.IPhysicalDevice.VK;
import RHI.ILogicalDevice.VK;
import RHI.ISwapChain.VK;

using namespace SIByL;

class SandboxApp :public SIByL::IApplication
{

};

auto SE_CREATE_APP() noexcept -> SIByL::IApplication*
{
#ifdef _DEBUG
	SE_TEST_EXEC_ALL();
#endif

	SandboxApp* app = new SandboxApp();
	WindowLayer* window_layer = app->addWindow({ SIByL::EWindowVendor::GLFW, 1280, 720, "Hello" });

	auto vk_context = CreateScope<RHI::IGraphicContextVK>();
	vk_context->attachWindow(window_layer->getWindow());
	auto dx12 = CreateScope<RHI::IPhysicalDeviceDX12>();
	auto vk_device = CreateScope<RHI::IPhysicalDeviceVK>(vk_context.get());
	auto vk_logical_device = CreateScope<RHI::ILogicalDeviceVK>(vk_device.get());
	auto vk_swapbuffer = CreateScope<RHI::ISwapChainVK>(vk_logical_device.get());
	return app;
}