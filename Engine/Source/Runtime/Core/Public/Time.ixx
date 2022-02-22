module;
#include <chrono>
export module Core.Time;

namespace SIByL
{
	inline namespace Core
	{
		export class Time
		{

		};

		export class Timer
		{
		public:
			auto reset() noexcept -> void;
			auto start() noexcept -> void;
			auto stop() noexcept -> void;
			auto tick() noexcept -> void;

			auto getFPS() noexcept -> unsigned int;
			auto getMsPF() noexcept -> double;
			auto getTotalTime() noexcept -> double;

		private:
			unsigned int framePerSecond = 0;
			double MsPerSecond = 0;

			std::chrono::microseconds deltaTime;
			std::chrono::microseconds pauseTime;
			std::chrono::system_clock::time_point baseTime;
			std::chrono::system_clock::time_point stopTime;
			std::chrono::system_clock::time_point prevTime;
			std::chrono::system_clock::time_point currTime;

			bool isStopped = true;
		};
	}
}