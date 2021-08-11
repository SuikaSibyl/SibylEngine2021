#pragma once

#ifdef SIBYL_PLATFORM_WINDOWS

extern SIByL::Application* SIByL::CreateApplication();

#include "Log.h"

int main(int argc, char** argv)
{
	SIByL::Log::Init();
	std::shared_ptr<spdlog::logger>&  logger = SIByL::Log::GetCoreLogger();
	SIByL_CORE_WARN("Log System starts running!");

	auto app = SIByL::CreateApplication();
	SIByL_CORE_WARN("Application starts running!");
	app->Run();
	delete app;
}

#endif // SIBYL_PLATFORM_WINDOWS
