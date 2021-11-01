#include "SIByL.h"
#include "InspectorPanel.h"

#include "imgui.h"
#include "imgui_internal.h"

#include "Editor/Utility/DrawProperty.h"
#include "EditorLayer.h"

#include "Sibyl/ECS/Core/Entity.h"

#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SPipeline.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"

namespace SIByL
{
	InspectorPanel::InspectorPanel()
	{

	}

	void InspectorPanel::OnImGuiRender()
	{
		ImGui::Begin("Inspector");
		if (m_State == State::EntityComponents && m_SelectEntity)
		{
			DrawComponents(m_SelectEntity);
		}
		else if (m_State == State::MaterialEditor)
		{
			SIByLEditor::DrawMaterial("Material", *m_SelectMaterial);
		}
		ImGui::End();
	}

	void InspectorPanel::SetSelectedEntity(const Entity& entity)
	{
		m_SelectEntity = entity;
		m_State = State::EntityComponents;
	}

	void InspectorPanel::SetSelectedMaterial(Ref<Material> material)
	{
		m_SelectMaterial = material;
		m_State = State::MaterialEditor;
	}

	template<typename T, typename UIFunction>
	static void DrawComponent(const std::string& name, Entity entity, UIFunction uiFunction)
	{
		const ImGuiTreeNodeFlags treeNodeFlags =
			ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_FramePadding |
			ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap;

		if (entity.HasComponent<T>())
		{
			ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

			ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4,4 });
			float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
			ImGui::Separator();
			bool open = ImGui::TreeNodeEx((void*)typeid(T).hash_code(), treeNodeFlags, name.c_str());
			ImGui::PopStyleVar();
			ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5);
			if (ImGui::Button("+", ImVec2{ lineHeight, lineHeight }))
			{
				ImGui::OpenPopup("ComponentSettings");
			}

			bool removeComponent = false;
			if (ImGui::BeginPopup("ComponentSettings"))
			{
				if (ImGui::MenuItem("Remove Component"))
					removeComponent = true;
				ImGui::EndPopup();
			}
			if (open)
			{
				T& component = entity.GetComponent<T>();
				uiFunction(component);
				ImGui::TreePop();
			}

			if (removeComponent)
				entity.RemoveComponent<T>();
		}
	}

	void InspectorPanel::DrawComponents(Entity entity)
	{
		const ImGuiTreeNodeFlags treeNodeFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowItemOverlap;

		if (entity.HasComponent<TagComponent>())
		{
			std::string& tag = entity.GetComponent<TagComponent>().Tag;

			char buffer[256];
			memset(buffer, 0, sizeof(buffer));
			strcpy_s(buffer, tag.c_str());
			if (ImGui::InputText(" ", buffer, sizeof(buffer)))
			{
				tag = std::string(buffer);
			}
		}

		DrawComponent<TransformComponent>("Transform", entity, [](auto& component)
			{
				SIByLEditor::DrawVec3Control("Translation", component.Translation);
				SIByLEditor::DrawVec3Control("Scale", component.Scale);
				SIByLEditor::DrawVec3Control("Rotation", component.EulerAngles);
			});

		DrawComponent<SpriteRendererComponent>("Sprite Renderer", entity, [](auto& component)
			{
				SIByLEditor::DrawMaterial("Material", *component.Material);
			});

		DrawComponent<MeshFilterComponent>("Mesh Filter", entity, [](auto& component)
			{
				SIByLEditor::DrawTriangleMeshSocket("Mesh", component);
			});

		DrawComponent<MeshRendererComponent>("Mesh Renderer", entity, [](auto& component)
			{
				for (const auto& iter : SRenderPipeline::SRenderContext::GetRenderPipeline()->GetDrawPassesNames())
				{
					ImGui::Text(iter.c_str());

					for (int i = 0; i < component.SubmeshNum; i++)
					{
						std::string slot = std::string("Material ") + std::to_string(i);
						SIByLEditor::DrawMeshRendererMaterialSocket(slot, iter, component, i);
					}

					ImGui::Separator();
				}
			});

		DrawComponent<LightComponent>("Light", entity, [](auto& component)
			{
				SIByLEditor::DrawLight("Light", component);
			});

		{
			ImGui::Separator();
			ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
			ImVec2 buttonSize(200, 30);

			ImGui::SetCursorPosX(contentRegionAvailable.x / 2 - 100);
			if (ImGui::Button("Add Component", buttonSize))
				ImGui::OpenPopup("AddComponent");

			if (ImGui::BeginPopup("AddComponent"))
			{
				if (ImGui::MenuItem("Transform"))
				{
					m_SelectEntity.AddComponent<TransformComponent>();
					ImGui::CloseCurrentPopup();
				}

				if (ImGui::MenuItem("SpriteRenderer"))
				{
					m_SelectEntity.AddComponent<SpriteRendererComponent>();
					ImGui::CloseCurrentPopup();
				}

				if (ImGui::MenuItem("MeshFilter"))
				{
					m_SelectEntity.AddComponent<MeshFilterComponent>();
					ImGui::CloseCurrentPopup();
				}

				if (ImGui::MenuItem("MeshRenderer"))
				{
					if (m_SelectEntity.HasComponent<MeshFilterComponent>())
					{
						MeshFilterComponent& meshFilter = m_SelectEntity.GetComponent<MeshFilterComponent>();
						UINT matNum = meshFilter.GetSubmeshNum();
						MeshRendererComponent& meshRenderer = m_SelectEntity.AddComponent<MeshRendererComponent>();
						meshRenderer.SetMaterialNums(matNum);
					}
					ImGui::CloseCurrentPopup();
				}

				if (ImGui::MenuItem("Light"))
				{
					m_SelectEntity.AddComponent<LightComponent>();
					ImGui::CloseCurrentPopup();
				}

				if (ImGui::MenuItem("SelfCollisionDetector"))
				{
					if (m_SelectEntity.HasComponent<MeshFilterComponent>())
					{
						MeshFilterComponent& meshFilter = m_SelectEntity.GetComponent<MeshFilterComponent>();
						SelfCollisionDetectorComponent& sc = m_SelectEntity.AddComponent<SelfCollisionDetectorComponent>();
						sc.Init(meshFilter);
					}
					ImGui::CloseCurrentPopup();
				}

				ImGui::EndPopup();
			}
		}
	}
}