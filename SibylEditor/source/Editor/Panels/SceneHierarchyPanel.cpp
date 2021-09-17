#include "SIByL.h"
#include "SceneHierarchyPanel.h"

#include "Sibyl/ECS/Scene/Scene.h"
#include "Sibyl/ECS/Components/Common/Tag.h"
#include "Sibyl/ECS/Core/Entity.h"
#include "Sibyl/ECS/Components/Components.h"

#include "imgui.h"
#include "imgui_internal.h"

#include "Editor/Utility/DrawProperty.h"
#include "EditorLayer.h"

namespace SIByL
{
	SceneHierarchyPanel::SceneHierarchyPanel(const Ref<Scene>& context)
	{
		SetContext(context);
	}

	void SceneHierarchyPanel::SetContext(const Ref<Scene>& context)
	{
		m_Context = context;
	}

	void SceneHierarchyPanel::OnImGuiRender()
	{
		{
			ImGui::Begin("Scene Hierarchy");

			m_Context->m_Registry.each([&](auto entityID)
				{
					Entity entity{ entityID, m_Context.get() };
					DrawEntityNode(entity);
				});

			if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
			{
				SIByLEditor::EditorLayer::s_InspectorPanel.m_SelectEntity = {};
			}

			// Right-click on blank space
			if (ImGui::BeginPopupContextWindow(0, 1, false))
			{
				if (ImGui::MenuItem("Create Empty Entity"))
					m_Context->CreateEntity("Empty Entity");

				ImGui::EndPopup();
			}
			ImGui::End();
		}
	}

	Entity SceneHierarchyPanel::GetSelectedEntity() const { return SIByLEditor::EditorLayer::s_InspectorPanel.m_SelectEntity; }

	void SceneHierarchyPanel::DrawEntityNode(Entity entity)
	{
		TagComponent& tag = entity.GetComponent<TagComponent>();

		ImGuiTreeNodeFlags flags = ((SIByLEditor::EditorLayer::s_InspectorPanel.m_SelectEntity == entity) ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow;
		flags |= ImGuiTreeNodeFlags_SpanAvailWidth;

		bool opened = ImGui::TreeNodeEx((void*)(uint64_t)(uint32_t)entity, flags, tag.Tag.c_str());
		if (ImGui::IsItemClicked())
		{
			SIByLEditor::EditorLayer::s_InspectorPanel.SetSelectedEntity(entity);
		}

		bool entityDeleted = false;
		// Right-click on blank space
		if (ImGui::BeginPopupContextItem())
		{
			if (ImGui::MenuItem("Delete Entity"))
				entityDeleted = true;

			ImGui::EndPopup();
		}

		if (opened)
		{
			ImGuiTreeNodeFlags flags = ((SIByLEditor::EditorLayer::s_InspectorPanel.m_SelectEntity == entity) ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
			bool opened = ImGui::TreeNodeEx((void*)1000, flags, tag.Tag.c_str());
			if (opened)
				ImGui::TreePop();
			ImGui::TreePop();
		}

		if (entityDeleted)
		{
			m_Context->DestroyEntity(entity);
			if (SIByLEditor::EditorLayer::s_InspectorPanel.m_SelectEntity == entity)
			{
				SIByLEditor::EditorLayer::s_InspectorPanel.SetSelectedEntity({});
			}
		}
	}

}