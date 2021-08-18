#include "SIByLpch.h"
#include "Application.h"
#include "Sibyl/Events/ApplicationEvent.h"
#include "Sibyl/Core/Input.h"
#include "glad/glad.h"

namespace SIByL
{
	Application* Application::s_Instance = nullptr;

	Application::Application()
	{
		SIByL_CORE_ASSERT(!s_Instance, "Application already exists!");
		s_Instance = this;

		m_Window = std::unique_ptr<Window>(Window::Create());
		m_Window->SetEventCallback(BIND_EVENT_FN(Application::OnEvent));

		PushOverlay(SIByL::ImGuiLayer::Create());
	}

	Application::~Application()
	{

	}

	void Application::Run()
	{
		while (m_Running)
		{
			// Update
			for (Layer* layer : m_LayerStack)
			{
				layer->OnUpdate();
			}

			// Draw
			m_Window->OnUpdate();
		}
	}

	void Application::OnEvent(Event& e)
	{
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<WindowCloseEvent>(BIND_EVENT_FN(Application::OnWindowClosed));

		//SIByL_CORE_TRACE("{0}", e);

		for (auto it = m_LayerStack.end(); it != m_LayerStack.begin();)
		{
			(*--it)->OnEvent(e);
			if (e.Handled)
				break;
		}
	}

	void Application::OnDraw()
	{
		for (auto layer : m_LayerStack)
		{
			layer->OnDraw();
		}
	}

	void Application::DrawImGui()
	{
		for (auto layer : m_LayerStack)
		{
			layer->OnDrawImGui();
		}
	}

	void Application::PushLayer(Layer* layer)
	{
		m_LayerStack.PushLayer(layer);
		layer->OnAttach();
	}

	void Application::PushOverlay(Layer* overlay)
	{
		m_LayerStack.PushOverlay(overlay);
		overlay->OnAttach();
	}

	bool Application::OnWindowClosed(WindowCloseEvent& e)
	{
		m_Running = false;
		return true;
	}
}