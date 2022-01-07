module;
#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
export module Core.Log;

namespace SIByL::Core
{
	class Logger
	{
	public:
		Logger();
		static auto instance() noexcept -> Logger&;

		std::shared_ptr<spdlog::logger>& getCoreLogger();
		std::shared_ptr<spdlog::logger>& getClientLogger();

	private:
		std::shared_ptr<spdlog::logger> coreLogger;
		std::shared_ptr<spdlog::logger> clientLogger;
	};
}

export template <class... Args> auto SE_CORE_TRACE(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getCoreLogger()->trace(std::forward<Args>(args)...); }
export template <class... Args> auto SE_CORE_INFO(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getCoreLogger()->info(std::forward<Args>(args)...); }
export template <class... Args> auto SE_CORE_DEBUG(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getCoreLogger()->debug(std::forward<Args>(args)...); }
export template <class... Args> auto SE_CORE_WARN(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getCoreLogger()->warn(std::forward<Args>(args)...); }
export template <class... Args> auto SE_CORE_ERROR(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getCoreLogger()->error(std::forward<Args>(args)...); }

export template <class... Args> auto SE_TRACE(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getClientLogger()->trace(std::forward<Args>(args)...); }
export template <class... Args> auto SE_INFO(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getClientLogger()->info(std::forward<Args>(args)...); }
export template <class... Args> auto SE_DEBUG(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getClientLogger()->debug(std::forward<Args>(args)...); }
export template <class... Args> auto SE_WARN(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getClientLogger()->warn(std::forward<Args>(args)...); }
export template <class... Args> auto SE_ERROR(Args&&... args) noexcept -> void { ::SIByL::Core::Logger::instance().getClientLogger()->error(std::forward<Args>(args)...); }
