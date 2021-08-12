#include "SIByLpch.h"
#include "ImGuiLayerOpenGL.h"
#include "Platform/OpenGL/ImGuiOpenGLRenderer.h"

namespace SIByL
{
#ifdef RENDER_API_OpenGL
	ImGuiLayer* ImGuiLayer::Create()
	{
		return new ImGuiLayerOpenGL();
	}
#endif

	void ImGuiLayerOpenGL::PlatformInit()
	{
		ImGui_ImplOpenGL3_Init("#version 410");
	}

	void ImGuiLayerOpenGL::NewFrameBegin()
	{
		ImGui_ImplOpenGL3_NewFrame();
	}

	void ImGuiLayerOpenGL::NewFrameEnd()
	{
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
}