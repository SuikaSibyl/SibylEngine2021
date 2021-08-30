#include "SIByLpch.h"
#include "Application.h"
#include "Input.h"
#include "Sibyl/Events/ApplicationEvent.h"
#include "Sibyl/Core/Input.h"
#include "glad/glad.h"
#include "Sibyl/Renderer/GraphicContext.h"
#include "Sibyl/Renderer/Renderer2D.h"
#include "Sibyl/Graphic/Geometry/Vertex.h"

namespace SIByL
{
	Application* Application::s_Instance = nullptr;

	Application::Application()
	{
		// Init Application
		// --------------------------------
		SIByL_CORE_ASSERT(!s_Instance, "Application already exists!");
		s_Instance = this;

		// Init: Window
		// --------------------------------
		m_Window = Window::Create();

		// Init: Input System
		// --------------------------------
		Input::Init();
		m_Window->SetEventCallback(BIND_EVENT_FN(Application::OnEvent));

		// Init: ImGui
		// --------------------------------
		PushOverlay(SIByL::ImGuiLayer::Create());

		// Init: Frame Timer
		// --------------------------------
		m_FrameTimer.reset(FrameTimer::Create());
		m_FrameTimer->Reset();
	}

	Application::~Application()
	{
		m_Window->GetGraphicContext()->GetSynchronizer()->ForceSynchronize();
		Input::Destroy();
		Renderer2D::Shutdown();
	}

	void Application::OnAwake()
	{
		// Awake: Render Objects
		// --------------------------------
		Ref<CommandList> cmdList = m_Window->GetGraphicContext()->GetCommandList();
		Ref<Synchronizer> synchronizer = m_Window->GetGraphicContext()->GetSynchronizer();

		synchronizer->StartFrame();
		cmdList->Restart();

		// ---------------------------------------

		// Init::Renderer2D
		// --------------------------------
		Renderer2D::Init();
		// Init::All Layers
		// --------------------------------
		for (Layer* layer : m_LayerStack)
			layer->OnInitResource();

		// ---------------------------------------
		cmdList->Execute();
		synchronizer->EndFrame();
		m_Window->GetGraphicContext()->GetSynchronizer()->ForceSynchronize();
	}

	void Application::Run()
	{
		while (m_Running)
		{
			m_FrameTimer->Tick();

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
		dispatcher.Dispatch<WindowResizeEvent>(BIND_EVENT_FN(Application::OnWindowResized));

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

	void Application::OnResourceDestroy()
	{
		for (auto layer : m_LayerStack)
		{
			layer->OnReleaseResource();
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

	bool Application::OnWindowResized(WindowResizeEvent& e)
	{
		if (e.GetWidth() == 0 || e.GetHeight() == 0)
		{
			m_IsMinimized = true;
			return false;
		}
		
		m_IsMinimized = false;
		m_Window->GetGraphicContext()->OnWindowResize(e.GetWidth(), e.GetHeight());

		return false;
	}
}