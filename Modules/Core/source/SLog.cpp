#include "SLog.h"
#include <stdarg.h>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include <spdlog/sinks/stdout_color_sinks.h>

namespace SIByL
{
	namespace Core
	{
		static std::shared_ptr<spdlog::logger> s_CoreLogger;
		static std::shared_ptr<spdlog::logger> s_ClientLogger;
		
		static void Init()
		{
			spdlog::set_pattern("%^[%T] %n:  %v%$");

			s_CoreLogger = spdlog::stdout_color_mt("CORE");
			s_CoreLogger->set_level(spdlog::level::trace);

			s_ClientLogger = spdlog::stdout_color_mt("CLIENT");
			s_ClientLogger->set_level(spdlog::level::trace);
		}

		static std::shared_ptr<spdlog::logger>& GetCoreLogger()
		{
			if (s_CoreLogger == nullptr) Init();
			return s_CoreLogger;
		}
		static std::shared_ptr<spdlog::logger>& GetClientLogger() { return s_ClientLogger; }


		void SLog::Core_Trace(int i, ...)
		{
			{
				std::cout << "hello";
			}
			/* Declare a va_list type variable */
			va_list myargs;

			/* Initialise the va_list variable with the ... after fmt */

			va_start(myargs, i);

			/* Forward the '...' to vprintf */
			GetCoreLogger()->trace(myargs);

			/* Clean up the va_list */
			va_end(myargs);
		}
	}
}