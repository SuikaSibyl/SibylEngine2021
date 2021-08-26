#pragma once

#include "Sibyl/Renderer/Synchronizer.h"

namespace SIByL
{
	class OpenGLSynchronizer : public Synchronizer
	{
	public:
		virtual void ForceSynchronize() override {}
		virtual void StartFrame() override {}
		virtual bool CheckFinish(UINT64 fence) override { return true; }
		virtual void EndFrame() override {}

	private:

	};
}