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

using namespace SIByL;

class SandboxApp :public SIByL::IApplication
{

};

auto SE_CREATE_APP() noexcept -> SIByL::IApplication*
{
#ifdef _DEBUG
	SE_TEST_EXEC_ALL();
#endif

	auto vk = CreateScope<RHI::IGraphicContextVK>();
	auto dx12 = CreateScope<RHI::IPhysicalDeviceDX12>();
	auto vk_device = CreateScope<RHI::IPhysicalDeviceVK>();
	auto vk_logical_device = CreateScope<RHI::ILogicalDeviceVK>(vk_device.get());

	SandboxApp* app = new SandboxApp();
	app->addWindow({ SIByL::EWindowVendor::GLFW, 1280, 720, "Hello" });
	//app->addWindow({ SIByL::EWindowVendor::GLFW, 1280, 720, "World" });
	return app;
}