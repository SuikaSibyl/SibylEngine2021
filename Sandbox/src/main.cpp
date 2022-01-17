#include <iostream>
#include <memory>
#include <EntryPoint.h>

import Core.Application;
import Core.Assert;
import Core.Test;
import Core.Window;
import Core.Enums;
import Core.SObject;
import RHI.GraphicContext.VK;
import RHI.IPhysicalDevice.DX12;
import RHI.IPhysicalDevice.VK;

class SandboxApp :public SIByL::IApplication
{

};

auto SE_CREATE_APP() noexcept -> SIByL::IApplication*
{
#ifdef _DEBUG
	SE_TEST_EXEC_ALL();
#endif
	SIByL::RHI::IGraphicContextVK* vk = SIByL::SFactory::create<SIByL::RHI::IGraphicContextVK>();
	SIByL::RHI::IPhysicalDeviceDX12* dx12 = SIByL::SFactory::create<SIByL::RHI::IPhysicalDeviceDX12>();
	SIByL::RHI::IPhysicalDeviceVK* vk_device = SIByL::SFactory::create<SIByL::RHI::IPhysicalDeviceVK>();

	SIByL::SFactory::destroy(vk_device);
	SIByL::SFactory::destroy(dx12);
	SIByL::SFactory::destroy(vk);

	//vk.checkValidationLayerSupport();
	SandboxApp* app = new SandboxApp();
	app->addWindow({ SIByL::EWindowVendor::GLFW, 1280, 720, "Hello" });
	//app->addWindow({ SIByL::EWindowVendor::GLFW, 1280, 720, "World" });
	return app;
}