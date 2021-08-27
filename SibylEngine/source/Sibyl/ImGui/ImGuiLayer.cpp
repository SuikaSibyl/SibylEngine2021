#include "SIByLpch.h"
#include "ImGuiLayer.h"

#include "imgui.h"
#include "glad/glad.h"

#include "GLFW/glfw3.h"
#include "Sibyl/Core/Application.h"

#include "Sibyl/Renderer/Renderer.h"
#include "Platform/OpenGL/ImGui/ImGuiLayerOpenGL.h"
#include "Platform/DirectX12/ImGui/ImGuiLayerDX12.h"

namespace SIByL
{
	ImGuiLayer* ImGuiLayer::Main = nullptr;

	ImGuiLayer::ImGuiLayer()
		:Layer("ImGuiLayer")
	{
		SIByL_CORE_ASSERT(!Main, "ImGuiLayer Already Exists!");
		Main = this;
	}
	
	void ImGuiLayer::OnReleaseResource()
	{
		//PlatformDestroy();
	}

	ImGuiLayer::~ImGuiLayer()
	{
		
	}

	ImGuiLayer* ImGuiLayer::Create()
	{
		switch (Renderer::GetRaster())
		{
		case RasterRenderer::OpenGL: return new ImGuiLayerOpenGL(); break;
		case RasterRenderer::DirectX12: return new ImGuiLayerDX12(); break;
		case RasterRenderer::CpuSoftware: return nullptr; break;
		case RasterRenderer::GpuSoftware: return nullptr; break;
		default: return nullptr; break;
		}
	}

	void ImGuiLayer::OnAttach()
	{
		// Setup Dear ImGui context
		IMGUI_CHECKVERSION();

		ImGui::CreateContext();
		ImGui::StyleColorsDark();

		ImGuiIO& io = ImGui::GetIO();
		io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
		io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;
		io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
		io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
		io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
		// Setup Platform/Renderer bindings
		PlatformInit();
	}

	void ImGuiLayer::OnDetach()
	{

	}

	void ImGuiLayer::OnUpdate()
	{
		ImGuiIO& io = ImGui::GetIO();
		Application& app = Application::Get();
		io.DisplaySize = ImVec2((float)app.GetWindow().GetWidth(), (float)app.GetWindow().GetHeight());

		float time = (float)glfwGetTime();
		io.DeltaTime = m_Time > 0.0f ? (time - m_Time) : (1.0f / 60.0f);

		NewFrameBegin();
		ImGui::NewFrame();

		app.DrawImGui();

		ImGui::Render();
	}

	void ImGuiLayer::OnDraw()
	{
		NewFrameEnd();
	}

	void ImGuiLayer::OnDrawImGui()
	{
		static bool show = true;
		ImGui::ShowDemoWindow(&show);
	}

	void ImGuiLayer::OnEvent(Event& event)
	{
		
	}
}