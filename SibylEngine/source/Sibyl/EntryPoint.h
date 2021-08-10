#pragma once

#ifdef SIBYL_PLATFORM_WINDOWS

extern SIByL::Application* SIByL::CreateApplication();

#include <memory>
#include <string>
#include "Log.h"

int main(int argc, char** argv)
{
	SIByL::Log::Init();
	std::shared_ptr<spdlog::logger>&  logger = SIByL::Log::GetCoreLogger();
	logger->warn("Hello");

	auto app = SIByL::CreateApplication();
	app->Run();
	delete app;
}

#endif // SIBYL_PLATFORM_WINDOWS
