#include "SLog.h"

#include "../vendor/spdlog/include/spdlog/sinks/stdout_color_sinks.h"

namespace SIByL
{
	namespace Core
	{
		std::shared_ptr<spdlog::logger> SLog::s_CoreLogger;
		std::shared_ptr<spdlog::logger> SLog::s_ClientLogger;
		
		void SLog::Init()
		{
			spdlog::set_pattern("%^[%T] %n:  %v%$");

			s_CoreLogger = spdlog::stdout_color_mt("CORE");
			s_CoreLogger->set_level(spdlog::level::trace);

			s_ClientLogger = spdlog::stdout_color_mt("CLIENT");
			s_ClientLogger->set_level(spdlog::level::trace);
		}

		std::shared_ptr<spdlog::logger>& SLog::GetCoreLogger()
		{
			if (s_CoreLogger == nullptr) Init();
			return s_CoreLogger;
		}

		std::shared_ptr<spdlog::logger>& SLog::GetClientLogger() {
			if (s_ClientLogger == nullptr) Init();
			return s_ClientLogger;
		}
	}
}