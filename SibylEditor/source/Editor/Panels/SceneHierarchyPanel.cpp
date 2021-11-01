#include "SIByL.h"
#include "SceneHierarchyPanel.h"

#include "Sibyl/ECS/Scene/Scene.h"
#include "Sibyl/ECS/Components/Common/Tag.h"
#include "Sibyl/ECS/Core/Entity.h"
#include "Sibyl/ECS/Components/Components.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"

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
		SRenderPipeline::SRenderContext::SetActiveScene(context);
	}

	void SceneHierarchyPanel::OnImGuiRender()
	{
		{
			ImGui::Begin("Scene Hierarchy");

			ImGui::BeginChild("abc");
			// Iterat all entities, draw those without parent
			m_Context->m_Registry.each([&](auto entityID)
				{
					Entity entity{ entityID, m_Context.get() };
					if (entity.GetComponent<TransformComponent>().GetParent() == 0)
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

			ImGui::EndChild();
			// Drop the entity on blank space
			if (ImGui::BeginDragDropTarget())
			{
				if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ENTITY"))
				{
					const uint64_t* uid = (uint64_t*)payload->Data;
					Entity child(*uid, m_Context.get());
					child.GetComponent<TransformComponent>().SetParent(0);
				}
				ImGui::EndDragDropTarget();
			}

			ImGui::End();
		}
	}

	Entity SceneHierarchyPanel::GetSelectedEntity() const { return SIByLEditor::EditorLayer::s_InspectorPanel.m_SelectEntity; }

	void SceneHierarchyPanel::DrawEntityNode(Entity entity)
	{
		ImGui::PushID(entity.GetUid());

		TagComponent& tag = entity.GetComponent<TagComponent>();
		std::vector<uint64_t>& children = entity.GetComponent<TransformComponent>().GetChildren();
		bool nochildren = children.size() == 0;

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
		// ------------------
		//  Begin DragDropSource()
		if (ImGui::BeginDragDropSource())
		{
			ImGui::Text(tag.Tag.c_str());
			uint64_t uid = entity.GetUid();
			ImGui::SetDragDropPayload("ENTITY", &uid, sizeof(uid));
			ImGui::EndDragDropSource();
		}
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ENTITY"))
			{
				const uint64_t* uid = (uint64_t*)payload->Data;
				Entity child(*uid, m_Context.get());
				child.GetComponent<TransformComponent>().SetParent(entity.GetUid());
			}
			ImGui::EndDragDropTarget();
		}

		if (opened)
		{
			for (int i = 0; i < children.size(); i++)
			{
				Entity child(children[i], m_Context.get());
				DrawEntityNode(child);
			}
			//ImGuiTreeNodeFlags flags = ((SIByLEditor::EditorLayer::s_InspectorPanel.m_SelectEntity == entity) ? ImGuiTreeNodeFlags_Selected : 0) | ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
			//bool opened = ImGui::TreeNodeEx((void*)1000, flags, tag.Tag.c_str());
			//if (opened)
			//	ImGui::TreePop();
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
		ImGui::PopID();
	}

}