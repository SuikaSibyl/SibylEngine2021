#pragma once

#include "../vendor/spdlog/include/spdlog/spdlog.h"
#include "../vendor/spdlog/include/spdlog/fmt/ostr.h"

namespace SIByL
{
	namespace Core
	{
		class SLog
		{
		public:			
			static std::shared_ptr<spdlog::logger>& GetCoreLogger();
			static std::shared_ptr<spdlog::logger>& GetClientLogger();

		private:
			static void Init();

			static std::shared_ptr<spdlog::logger> s_CoreLogger;
			static std::shared_ptr<spdlog::logger> s_ClientLogger;
		};
	}
}

#define S_CORE_TRACE(...)	::SIByL::Core::SLog::GetCoreLogger()->trace(__VA_ARGS__)
#define S_CORE_INFO(...)	::SIByL::Core::SLog::GetCoreLogger()->info(__VA_ARGS__)
#define S_CORE_DEBUG(...)	::SIByL::Core::SLog::GetCoreLogger()->debug(__VA_ARGS__)
#define S_CORE_WARN(...)	::SIByL::Core::SLog::GetCoreLogger()->warn(__VA_ARGS__)
#define S_CORE_ERROR(...)	::SIByL::Core::SLog::GetCoreLogger()->error(__VA_ARGS__)

#define S_CLIENT_TRACE(...)	::SIByL::Core::SLog::GetClientLogger()->trace(__VA_ARGS__)
#define S_CLIENT_INFO(...)		::SIByL::Core::SLog::GetClientLogger()->info(__VA_ARGS__)
#define S_CLIENT_DEBUG(...)		::SIByL::Core::SLog::GetClientLogger()->debug(__VA_ARGS__)
#define S_CLIENT_WARN(...)		::SIByL::Core::SLog::GetClientLogger()->warn(__VA_ARGS__)
#define S_CLIENT_ERROR(...)	::SIByL::Core::SLog::GetClientLogger()->error(__VA_ARGS__)
