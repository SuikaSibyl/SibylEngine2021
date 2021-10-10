#include "SIByL.h"
#include "ViewportPanel.h"

#include "EditorLayer.h"
#include "Sibyl/Basic/Utils/MathUtils.h"
#include "Sibyl/ImGui/ImGuiUtility.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"

namespace SIByL
{
	void ViewportPanel::SetFrameBuffer(Ref<FrameBuffer> framebuffer)
	{
		m_FrameBuffer = framebuffer;
	}

	void ViewportPanel::SetCamera(Ref<Camera> camera)
	{
		m_Camera = camera;
	}

	const glm::vec2& ViewportPanel::GetViewportSize()
	{
		return m_ViewportSize;
	}
	
	bool ViewportPanel::IsViewportFocusd()
	{
		return m_ViewportFocused;
	}

	bool ViewportPanel::IsViewportHovered()
	{
		return m_ViewportHoverd;
	}

	void ViewportPanel::OnImGuiRender()
	{
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2({ 0,0 }));
		ImGui::Begin("Viewport", 0, ImGuiWindowFlags_MenuBar);
		
		ImGuiDrawMenu();

		m_ViewportFocused = ImGui::IsWindowFocused();
		m_ViewportHoverd = ImGui::IsWindowHovered();
		Application::Get().GetImGuiLayer()->SetBlockEvents(!m_ViewportFocused && !m_ViewportHoverd);

		ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
		if (m_ViewportSize != *((glm::vec2*)&viewportPanelSize))
		{
			// Viewport Change Size
			m_ViewportSize = { viewportPanelSize.x, viewportPanelSize.y };
			m_FrameBuffer->Resize(viewportPanelSize.x, viewportPanelSize.y);
			m_Camera->Resize(viewportPanelSize.x, viewportPanelSize.y);
		}

		//unsigned int textureID = m_FrameBuffer->GetColorAttachment();
		ImGui::DrawImage((void*)m_FrameBuffer->GetColorAttachment(m_FrameBufferIndex), ImVec2{
			viewportPanelSize.x,
			viewportPanelSize.y });

		Entity selectedEntity = SIByLEditor::EditorLayer::s_SceneHierarchyPanel.GetSelectedEntity();
		if (selectedEntity && GizmoType != -1)
		{
			ImGuizmo::SetOrthographic(false);
			ImGuizmo::SetDrawlist();

			float windowWidth = (float)ImGui::GetWindowWidth();
			float windowHeight = (float)ImGui::GetWindowHeight();
			ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, windowWidth, windowHeight);

			glm::mat4 cameraView = m_Camera->GetViewMatrix();
			glm::mat4 cameraProj = m_Camera->GetProjectionMatrix();

			// Entity transform
			auto& tc = selectedEntity.GetComponent<TransformComponent>();
			glm::mat4 transform = tc.GetTransform();

			// Snapping 
			bool snap = Input::IsKeyPressed(SIByL_KEY_LEFT_CONTROL);
			float snapValue = 0.5f;
			if (GizmoType == ImGuizmo::OPERATION::ROTATE)
				snapValue = 45.0f;

			float snapValues[3] = { snapValue, snapValue ,snapValue };


			ImGuizmo::Manipulate(glm::value_ptr(cameraView), glm::value_ptr(cameraProj),
				ImGuizmo::OPERATION(GizmoType), ImGuizmo::LOCAL, glm::value_ptr(transform),
				nullptr, snap ? snapValues : nullptr);

			if (ImGuizmo::IsUsing())
			{
				glm::vec3 translation, rotation, scale;
				DecomposeTransform(transform, translation, rotation, scale);
				glm::vec3 deltaRotation = rotation - tc.EulerAngles;
				tc.Translation = translation;
				tc.Scale = scale;
				tc.EulerAngles += deltaRotation;
				transform = tc.GetTransform();
			}
		}

		if (ImGui::BeginDragDropTarget())
		{
			const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE");

			ImGui::EndDragDropTarget();
		}

		ImGui::End();
		ImGui::PopStyleVar();
	}

	bool ViewportPanel::OnKeyPressed(KeyPressedEvent& e)
	{
		if (e.GetKeyCode() == SIByL_KEY_Q)
		{
			if (!Input::IsMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				GizmoType = -1;
		}
		else if (e.GetKeyCode() == SIByL_KEY_W)
		{
			if (!Input::IsMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				GizmoType = 0;
		}
		else if (e.GetKeyCode() == SIByL_KEY_E)
		{
			if (!Input::IsMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				GizmoType = 1;
		}
		else if (e.GetKeyCode() == SIByL_KEY_R)
		{
			if (!Input::IsMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				GizmoType = 2;
		}

		return true;
	}

	static std::string ColorAttachmentTag[] =
	{
		"Attachment 0",
		"Attachment 1",
		"Attachment 2",
		"Attachment 3",
	};

	void ViewportPanel::DrawFrameBufferItem(const std::string& name, Ref<FrameBuffer> frameBuffer)
	{
		if (ImGui::BeginMenu(name.c_str()))
		{
			ImGui::MenuItem("(Attachment)", NULL, false, false);

			for (int i = 0; i < frameBuffer->CountColorAttachment(); i++)
			{
				if (ImGui::MenuItem(ColorAttachmentTag[i].c_str()))
				{
					m_FrameBuffer = frameBuffer;
					m_FrameBufferIndex = i;
				}
			}

			ImGui::EndMenu();
		}
	}

	void ViewportPanel::ImGuiDrawMenu()
	{
		ImGui::PushItemWidth(ImGui::GetFontSize() * -12);

		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("Choose Buffer"))
			{
				ImGui::MenuItem("(Frame Buffer)", NULL, false, false);

				for (auto& iter : Library<FrameBuffer>::Mapper)
				{
					DrawFrameBufferItem(iter.first, iter.second);
				}

				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}

		ImGui::PopItemWidth();
	}
}
