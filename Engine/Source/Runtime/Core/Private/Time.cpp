module;
#include <chrono>
module Core.Time;

namespace SIByL
{
	auto Timer::reset() noexcept -> void
	{
		currTime = std::chrono::system_clock::now();
		baseTime = currTime;
		
		isStopped = false;
	}

	auto Timer::start() noexcept -> void
	{
		if (isStopped)
		{
			currTime = std::chrono::system_clock::now();
			pauseTime += std::chrono::duration_cast<std::chrono::microseconds>(currTime - stopTime);
			baseTime = currTime;
			isStopped = false;
		}
	}

	auto Timer::stop() noexcept -> void
	{
		if (!isStopped)
		{
			currTime = std::chrono::system_clock::now();
			stopTime = currTime;
			isStopped = true;
		}
	}

	auto Timer::tick() noexcept -> void
	{
		if (isStopped)
		{
			deltaTime = std::chrono::microseconds(0);
			return;
		}

		currTime = std::chrono::system_clock::now();
		deltaTime = std::chrono::duration_cast<std::chrono::microseconds>(currTime - prevTime);
		prevTime = currTime;
	}

	auto Timer::getFPS() noexcept -> unsigned int
	{
		return 1000.0 / (0.001 * deltaTime.count());
	}

	auto Timer::getMsPF() noexcept -> double
	{
		return (0.001 * deltaTime.count());
	}

	auto Timer::getTotalTime() noexcept -> double
	{
		return (double)0.001 * (std::chrono::duration_cast<std::chrono::microseconds>(currTime - baseTime).count());
	}

	auto Timer::getTotalTimeSeconds() noexcept -> double
	{
		return (double)0.001 * getTotalTime();
	}
}