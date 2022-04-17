module;
#include <cstdint>
#include <imgui.h>
export module Editor.Viewport;
import Editor.Widget;
import Editor.ImImage;

namespace SIByL::Editor
{
	export struct Viewport :public Widget
	{
		virtual auto onDrawGui() noexcept -> void override;
		auto getWidth() noexcept -> uint32_t { return viewportPanelSize.x; }
		auto getHeight() noexcept -> uint32_t { return viewportPanelSize.y; }
		auto bindImImage(ImImage* image) noexcept -> void { bindedImage = image; }

		ImImage* bindedImage;
		ImVec2 viewportPanelSize = { 1280,720 };
	};

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
}