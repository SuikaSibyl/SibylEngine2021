#pragma once

#include "Core.h"

namespace SIByL
{
	class SIByL_API Application
	{
	public:
		Application();
		virtual ~Application();

		void Run();
	};

	// Defined in client
	Application* CreateApplication();
}