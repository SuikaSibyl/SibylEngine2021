#pragma once

#include "Sibyl/Core/FrameTimer.h"

namespace SIByL
{
	class WindowsFrameTimer :public FrameTimer
	{
	public:
		WindowsFrameTimer();

		virtual void Reset() override;
		virtual void Start() override;
		virtual void Stop() override;
		virtual void Tick() override;

	private:

	};
}