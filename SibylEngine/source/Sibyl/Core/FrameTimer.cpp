#include "SIByLpch.h"
#include "FrameTimer.h"

#include "Platform/Windows/Window/WindowsFrameTimer.h"

namespace SIByL
{
	FrameTimer* FrameTimer::Create()
	{
		// Currently Only Support Windows
		return new WindowsFrameTimer();
	}

	FrameTimer::FrameTimer():
		m_SencondsPerCount(0.0), m_DeltaTime(-1.0), m_BaseTime(0),
		m_PauseTime(0), m_StopTime(0), m_PrevTime(0), m_CurrentTime(0), m_IsStoped(false)
	{
	}
	float FrameTimer::TotalTime() const
	{
		if (m_IsStoped)
		{
			return (float)((m_StopTime - m_PauseTime - m_BaseTime) * m_SencondsPerCount);
		}
		else
		{
			return (float)((m_CurrentTime - m_PauseTime - m_BaseTime) * m_SencondsPerCount);
		}
	}

	void FrameTimer::RefreshFPS()
	{
		static float timeElapsed = 0.0f;
		m_FramePerSecond++;

		if (TotalTime() - timeElapsed >= 1.0f)
		{
			m_FPS = (float)m_FramePerSecond;
			m_MsPF = 1000.0f / m_FPS;
			m_FramePerSecond = 0;
			timeElapsed += 1.0f;
		}
	}
}