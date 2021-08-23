#pragma once

namespace SIByL
{
	class FrameTimer
	{
	public:
		static FrameTimer* Create();
		FrameTimer();

		float TotalTime()const;
		float DeltaTime()const { return (float)m_DeltaTime; }
		bool IsStoped() { return m_IsStoped; }

		virtual void Reset() = 0;
		virtual void Start() = 0;
		virtual void Stop() = 0;
		virtual void Tick() = 0;

	protected:
		double m_SencondsPerCount;
		double m_DeltaTime;

		__int64 m_BaseTime;
		__int64 m_PauseTime;	
		__int64 m_StopTime;
		__int64 m_PrevTime;
		__int64 m_CurrentTime;

		bool m_IsStoped;
	};
}