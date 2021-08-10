#pragma once

#include "Core.h"
#include "spdlog/spdlog.h"
#include <iostream>
#include <memory>

namespace SIByL
{
	class SIByL_API Log
	{
	public:
		static void Init();

		static void Logg();

		static int i;

		static std::shared_ptr<spdlog::logger>& GetCoreLogger();
		static std::shared_ptr<spdlog::logger>& GetClientLogger() { return s_ClientLogger; }

	private:
		static std::shared_ptr<spdlog::logger> s_CoreLogger;
		static std::shared_ptr<spdlog::logger> s_ClientLogger;
	};
}

#define SIByL_CORE_TRACE(...)	::SIByL::Log::GetCoreLogger()->trace(__VA_ARGS__)
#define SIByL_CORE_INFO(...)	::SIByL::Log::GetCoreLogger()->info(__VA_ARGS__)
#define SIByL_CORE_WARN(...)	::SIByL::Log::GetCoreLogger()->warn(__VA_ARGS__)
#define SIByL_CORE_ERROR(...)	::SIByL::Log::GetCoreLogger()->error(__VA_ARGS__)
#define SIByL_CORE_FATAL(...)	::SIByL::Log::GetCoreLogger()->fatal(__VA_ARGS__)

#define SIByL_APP_TRACE(...)	::SIByL::Log::GetClientLogger()->trace(__VA_ARGS__)
#define SIByL_APP_INFO(...)		::SIByL::Log::GetClientLogger()->info(__VA_ARGS__)
#define SIByL_APP_WARN(...)		::SIByL::Log::GetClientLogger()->warn(__VA_ARGS__)
#define SIByL_APP_ERROR(...)	::SIByL::Log::GetClientLogger()->error(__VA_ARGS__)
#define SIByL_APP_FATAL(...)	::SIByL::Log::GetClientLogger()->fatal(__VA_ARGS__)