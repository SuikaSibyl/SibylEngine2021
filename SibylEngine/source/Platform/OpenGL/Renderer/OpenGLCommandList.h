#pragma once

#include "Sibyl/Renderer/CommandList.h"

namespace SIByL
{
	class OpenGLCommandList :public CommandList
	{
	public:
		virtual void Restart() override {}
		virtual void Execute() override {}

	private:

	};
}