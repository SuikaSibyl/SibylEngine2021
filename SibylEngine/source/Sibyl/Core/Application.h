#pragma once

#include "Core.h"
#include "Window.h"
#include "Sibyl/Events/ApplicationEvent.h"
#include "Sibyl/Events/KeyEvent.h"
#include "Sibyl/Events/MouseEvent.h"
#include "Sibyl/Core/LayerStack.h"

namespace SIByL
{
	class SIByL_API Application
	{
	public:
		Application();
		virtual ~Application();

		void Run();
		void OnEvent(Event& e);

		void PushLayer(Layer* layer);
		void PushOverlay(Layer* overlay);

		inline Window& GetWindow() { return *m_Window; }
		inline static Application& Get() { return *s_Instance; }

	private:
		bool OnWindowClosed(WindowCloseEvent& e);

	private:
		std::unique_ptr<Window> m_Window;
		bool m_Running = true;
		LayerStack m_LayerStack;

		static Application* s_Instance;
	};

	// Defined in client
	Application* CreateApplication();
}