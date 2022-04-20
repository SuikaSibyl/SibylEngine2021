module;
#include <cstdint>
#include <imgui.h>
export module Editor.Viewport;
import Core.Window;
import Core.Time;
import GFX.Transform;
import Editor.Widget;
import Editor.ImImage;
import Editor.CameraController;

namespace SIByL::Editor
{
	export struct Viewport :public Widget
	{
		Viewport(WindowLayer* window_layer, Timer* timer);

		virtual auto onDrawGui() noexcept -> void override;
		virtual auto onUpdate() noexcept -> void override;

		auto getWidth() noexcept -> uint32_t { return viewportPanelSize.x; }
		auto getHeight() noexcept -> uint32_t { return viewportPanelSize.y; }
		auto bindImImage(ImImage* image) noexcept -> void { bindedImage = image; }

		GFX::Transform cameraTransform;
		ImImage* bindedImage;
		ImVec2 viewportPanelSize = { 1280,720 };
		SimpleCameraController cameraController;
	};

	Viewport::Viewport(WindowLayer* window_layer, Timer* timer)
		:cameraController(window_layer->getWindow()->getInput(), timer) 
	{ 
		cameraController.bindTransform(&cameraTransform);
	}

	auto Viewport::onDrawGui() noexcept -> void
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2({ 0,0 }));
		ImGui::Begin("Viewport", 0, ImGuiWindowFlags_MenuBar);

		//m_ViewportFocused = ImGui::IsWindowFocused();
		//m_ViewportHoverd = ImGui::IsWindowHovered();
		//Application::get().GetImGuiLayer()->SetBlockEvents(!m_ViewportFocused && !m_ViewportHoverd);

		ImVec2 viewport_panel_size = ImGui::GetContentRegionAvail();
		if ((viewportPanelSize.x != viewport_panel_size.x) || (viewportPanelSize.y != viewport_panel_size.y))
		{
			// Viewport Change Size
			viewportPanelSize = viewport_panel_size;
		//	SRenderPipeline::SRenderContext::SetScreenSize({ viewportPanelSize.x, viewportPanelSize.y });
		//	FrameBufferLibrary::ResizeAll(viewportPanelSize.x, viewportPanelSize.y);
		//	m_Camera->Resize(viewportPanelSize.x, viewportPanelSize.y);
		}

		if (bindedImage)
		{
			ImGui::Image(
				bindedImage->getImTextureID(),
				{ 1280,720 }, 
				{ 0,0 }, { 1, 1 });
		}
		//unsigned int textureID = m_FrameBuffer->GetColorAttachment();
		//ImGui::DrawImage(
		//	(void*)m_FrameBuffer->GetColorAttachment(m_FrameBufferIndex), 
		//	ImVec2{viewportPanelSize.x,
		//	viewportPanelSize.y });

		ImGui::End();
		ImGui::PopStyleVar();
	}

	auto Viewport::onUpdate() noexcept -> void
	{
		cameraController.onUpdate();
	}

}