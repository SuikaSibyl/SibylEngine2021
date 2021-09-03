#include "SIByLpch.h"
#include "Application.h"
#include "Input.h"
#include "Sibyl/Core/Events/ApplicationEvent.h"
#include "Sibyl/Core/Input.h"
#include "glad/glad.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/GraphicContext.h"
#include "Sibyl/Graphic/Core/Renderer/Renderer2D.h"
#include "Sibyl/Graphic/Core/Geometry/Vertex.h"

#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandList.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandQueue.h"


namespace SIByL
{
	Application* Application::s_Instance = nullptr;

	Application::Application(const std::string& name)
	{
		PROFILE_SCOPE_FUNCTION();
		// Init Application
		// --------------------------------
		SIByL_CORE_ASSERT(!s_Instance, "Application already exists!");
		s_Instance = this;

		// Init: Window
		// --------------------------------
		m_Window = Window::Create(WindowProps(name));

		// Init: Input System
		// --------------------------------
		Input::Init();
		m_Window->SetEventCallback(BIND_EVENT_FN(Application::OnEvent));

		// Init: ImGui
		// --------------------------------
		PushOverlay(m_ImGuiLayer = SIByL::ImGuiLayer::Create());

		// Init: Frame Timer
		// --------------------------------
		m_FrameTimer.reset(FrameTimer::Create());
		m_FrameTimer->Reset();
	}

	Application::~Application()
	{
		PROFILE_SCOPE_FUNCTION();
		m_Window->GetGraphicContext()->GetSynchronizer()->ForceSynchronize();
		Input::Destroy();
		Renderer2D::Shutdown();
	}

	void Application::OnAwake()
	{
		PROFILE_SCOPE_FUNCTION();
		// Awake: Render Objects
		// --------------------------------
		m_Window->GetGraphicContext()->StartCommandList();

		// ---------------------------------------

		// Init::Renderer2D
		// --------------------------------
		Renderer2D::Init();
		// Init::All Layers
		// --------------------------------
		for (Layer* layer : m_LayerStack)
			layer->OnInitResource();

		// ---------------------------------------
		m_Window->GetGraphicContext()->EndCommandList();
	}

	void Application::Run()
	{
		PROFILE_SCOPE_FUNCTION();

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
		PROFILE_SCOPE_FUNCTION();

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
		PROFILE_SCOPE_FUNCTION();

		for (auto layer : m_LayerStack)
		{
			layer->OnDraw();
		}
	}

	void Application::OnResourceDestroy()
	{
		PROFILE_SCOPE_FUNCTION();

		for (auto layer : m_LayerStack)
		{
			layer->OnReleaseResource();
		}
	}
	void Application::DrawImGui()
	{
		PROFILE_SCOPE_FUNCTION();
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
		PROFILE_SCOPE_FUNCTION();

		m_Running = false;
		return true;
	}

	bool Application::OnWindowResized(WindowResizeEvent& e)
	{
		PROFILE_SCOPE_FUNCTION();

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