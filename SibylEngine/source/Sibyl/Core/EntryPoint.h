#pragma once

#ifdef SIBYL_PLATFORM_WINDOWS

extern SIByL::Application* SIByL::CreateApplication();

#include "Log.h"

int main(int argc, char** argv)
{
	PROFILE_BEGIN_SESSION("Startup", "SIByL-Startup.json");
	SIByL::Log::Init();
	std::shared_ptr<spdlog::logger>&  logger = SIByL::Log::GetCoreLogger();
	SIByL_CORE_WARN("Log System is Working!");
	
	auto app = SIByL::CreateApplication();
	SIByL_CORE_WARN("Application Awake!");
	app->OnAwake();
	PROFILE_END_SESSION();

	PROFILE_BEGIN_SESSION("Runtime", "SIByL-Runtime.json");
	SIByL_CORE_WARN("Application Run!");
	app->Run();
	PROFILE_END_SESSION();

	PROFILE_BEGIN_SESSION("Shutdown", "SIByL-Shutdown.json");
	delete app;
	PROFILE_END_SESSION();

}

#endif // SIBYL_PLATFORM_WINDOWS
