#pragma once

#include "Core.h"
#include "Window.h"
#include "Sibyl/Events/ApplicationEvent.h"
#include "Sibyl/Events/KeyEvent.h"
#include "Sibyl/Events/MouseEvent.h"

namespace SIByL
{
	class SIByL_API Application
	{
	public:
		Application();
		virtual ~Application();

		void Run();
		void OnEvent(Event& e);

	private:
		bool OnWindowClosed(WindowCloseEvent& e);

	private:
		std::unique_ptr<Window> m_Window;
		bool m_Running = true;
	};

	// Defined in client
	Application* CreateApplication();
}