#pragma once

#include "Sibyl/Renderer/Synchronizer.h"

namespace SIByL
{
	class OpenGLSynchronizer : public Synchronizer
	{
	public:
		virtual void ForceSynchronize() override {}

	private:

	};
}