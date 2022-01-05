#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <spdlog/sinks/stdout_color_sinks.h>
module Core.Log;

import <memory>;

namespace SIByL::Core
{
	Logger::Logger()
	{
		spdlog::set_pattern("%^[%T] %n:  %v%$");

		coreLogger = spdlog::stdout_color_mt("CORE");
		coreLogger->set_level(spdlog::level::trace);

		clientLogger = spdlog::stdout_color_mt("APP");
		clientLogger->set_level(spdlog::level::trace);
	}

	auto Logger::instance() noexcept -> Logger&
	{
		static Logger logger;
		return logger;
	}

	std::shared_ptr<spdlog::logger>& Logger::getCoreLogger()
	{
		return coreLogger;
	}

	std::shared_ptr<spdlog::logger>& Logger::getClientLogger() {
		return clientLogger;
	}
}