#include <iostream>
#include <memory>
#include <EntryPoint.h>

import Core.Application;
import Core.Assert;
import Core.Test;
import Core.Window;
import Core.Enums;
import Core.GraphicContext.VK;

class SandboxApp :public SIByL::IApplication
{

};

auto SE_CREATE_APP() noexcept -> SIByL::IApplication*
{
#ifdef _DEBUG
	SE_TEST_EXEC_ALL();
#endif

	SandboxApp* app = new SandboxApp();
	app->addWindow({ SIByL::EWindowVendor::GLFW, 1280, 720, "Hello" });
	app->addWindow({ SIByL::EWindowVendor::GLFW, 1280, 720, "World" });
	return app;
}