#include "Log.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace SIByL
{
	std::shared_ptr<spdlog::logger> Log::s_CoreLogger;
	std::shared_ptr<spdlog::logger> Log::s_ClientLogger;

	int Log::i;

	void Log::Logg()
	{

	}
	std::shared_ptr<spdlog::logger>& Log::GetCoreLogger()
	{ 
		return s_CoreLogger; 
	}

	void Log::Init()
	{		

		spdlog::set_pattern("%^[%T] %n:  %v%$");
		s_CoreLogger = spdlog::stdout_color_mt("Suika Engine");
		s_CoreLogger->set_level(spdlog::level::trace);

		s_ClientLogger = spdlog::stdout_color_mt("Suika App");
		s_ClientLogger->set_level(spdlog::level::trace);
	}
}