#include "SIByLpch.h"
#include "ImGuiLayerOpenGL.h"

#include "GLFW/glfw3.h"
#include "Platform/OpenGL/Window/GLFWWindow.h"
#include "Platform/OpenGL/ImGui/ImGuiOpenGLRenderer.h"
#include "Platform/OpenGL/ImGui/ImGuiGLFWRenderer.h"

namespace SIByL
{
#ifdef RENDER_API_OpenGL
	ImGuiLayer* ImGuiLayer::Create()
	{
		return new ImGuiLayerOpenGL();
	}

	void ImGuiLayer::OnDrawAdditionalWindows()
	{
		ImGuiIO& io = ImGui::GetIO();
		// Update and Render additional Platform Windows
		if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
		{
			GLFWwindow* backup_current_context = glfwGetCurrentContext();
			ImGui::UpdatePlatformWindows();
			ImGui::RenderPlatformWindowsDefault();
			glfwMakeContextCurrent(backup_current_context);
		}
	}
#endif

	void ImGuiLayerOpenGL::PlatformInit()
	{
		ImGui_ImplGlfw_InitForOpenGL((GLFWwindow*)GLFWWindow::Get()->GetNativeWindow(), true);
		ImGui_ImplOpenGL3_Init("#version 410");
	}

	void ImGuiLayerOpenGL::NewFrameBegin()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
	}

	void ImGuiLayerOpenGL::NewFrameEnd()
	{
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	void ImGuiLayerOpenGL::PlatformDestroy()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext();
	}
}