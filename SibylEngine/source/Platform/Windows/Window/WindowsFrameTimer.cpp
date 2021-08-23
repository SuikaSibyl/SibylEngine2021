#include "SIByLpch.h"
#include "WindowsFrameTimer.h"

namespace SIByL
{
	WindowsFrameTimer::WindowsFrameTimer()
	{
		// Initialize SecondPerCount according to system
		__int64 countsPerSec;
		QueryPerformanceFrequency((LARGE_INTEGER*)&countsPerSec);
		m_SencondsPerCount = 1.0 / (double)countsPerSec;
	}

	void WindowsFrameTimer::Reset()
	{
		__int64 currTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&currTime);

		m_BaseTime = currTime;
		m_PrevTime = currTime;
		m_StopTime = 0;
		m_IsStoped = false;
	}

	void WindowsFrameTimer::Start()
	{
		__int64 startTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&startTime);
		if (m_IsStoped)
		{
			m_PauseTime += (startTime - m_StopTime);
			m_PrevTime = startTime;
			m_StopTime = 0;
			m_IsStoped = false;
		}
	}

	void WindowsFrameTimer::Stop()
	{
		if (!m_IsStoped)
		{
			__int64 currTime;
			QueryPerformanceCounter((LARGE_INTEGER*)&currTime);
			m_StopTime = currTime;
			m_IsStoped = true;
		}
	}

	void WindowsFrameTimer::Tick()
	{
		if (m_IsStoped)
		{
			// If it's stopped now ...
			m_DeltaTime = 0.0;
			return;
		}

		// Calc the current time
		__int64 currentTime;
		QueryPerformanceCounter((LARGE_INTEGER*)&currentTime);
		m_CurrentTime = currentTime;
		m_DeltaTime = (m_CurrentTime - m_PrevTime) * m_SencondsPerCount;
		m_PrevTime = m_CurrentTime;
		if (m_DeltaTime < 0)
		{
			m_DeltaTime = 0;
		}

		// Recalculate FPS
		RefreshFPS();
	}
}