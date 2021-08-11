#include "SIByLpch.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace SIByL
{
	std::shared_ptr<spdlog::logger> Log::s_CoreLogger;
	std::shared_ptr<spdlog::logger> Log::s_ClientLogger;

	std::shared_ptr<spdlog::logger>& Log::GetCoreLogger()
	{ 
		return s_CoreLogger; 
	}

	void Log::Init()
	{
		spdlog::set_pattern("%^[%T] %n:  %v%$");

		s_CoreLogger = spdlog::stdout_color_mt("SIByL");
		s_CoreLogger->set_level(spdlog::level::trace);

		s_ClientLogger = spdlog::stdout_color_mt("Client");
		s_ClientLogger->set_level(spdlog::level::trace);
	}
}