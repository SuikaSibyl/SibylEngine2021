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

class SandboxApp :public SIByL::IApplication
{

};

auto SE_CREATE_APP() noexcept -> SIByL::IApplication*
{
#ifdef _DEBUG
	SE_TEST_EXEC_ALL();
#endif
	SIByL::RHI::IGraphicContextVK vk;
	vk.initVulkan();
	vk.cleanUp();
	SIByL::RHI::IPhysicalDeviceDX12* dx12 = SIByL::SFactory::create<SIByL::RHI::IPhysicalDeviceDX12>();

	//vk.checkValidationLayerSupport();
	SandboxApp* app = new SandboxApp();
	app->addWindow({ SIByL::EWindowVendor::GLFW, 1280, 720, "Hello" });
	//app->addWindow({ SIByL::EWindowVendor::GLFW, 1280, 720, "World" });
	return app;
}