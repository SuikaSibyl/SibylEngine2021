#pragma once

#ifdef SIBYL_PLATFORM_WINDOWS

extern SIByL::Application* SIByL::CreateApplication();

#include "Log.h"

int main(int argc, char** argv)
{
	SIByL::Log::Init();
	std::shared_ptr<spdlog::logger>&  logger = SIByL::Log::GetCoreLogger();
	SIByL_CORE_WARN("Log System is Working!");

	auto app = SIByL::CreateApplication();
	SIByL_CORE_WARN("Application Awake!");
	app->OnAwake();
	SIByL_CORE_WARN("Application Run!");
	app->Run();
	delete app;
}

#endif // SIBYL_PLATFORM_WINDOWS
