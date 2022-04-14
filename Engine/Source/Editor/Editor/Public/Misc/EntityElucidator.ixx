module;
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <imgui.h>
#include <imgui_internal.h>
#include "entt.hpp"
#include <map>
export module Editor.EntityElucidator;
import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;
import GFX.Mesh;

namespace SIByL::Editor
{
	export struct EntityElucidator
	{
		static auto drawInspector(ECS::Entity entity) -> void;
	};

	template<typename T, typename UIFunction>
	void drawComponent(ECS::Entity entity, const std::string& name, UIFunction uiFunction)
	{
		const ImGuiTreeNodeFlags treeNodeFlags =
			ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_FramePadding |
			ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap;

		if (entity.hasComponent<T>())
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
				T& component = entity.getComponent<T>();
				uiFunction(component);
				ImGui::TreePop();
			}

			if (removeComponent)
				entity.removeComponent<T>();
		}
	}

	auto EntityElucidator::drawInspector(ECS::Entity entity) -> void
	{
		// Draw Tag Component
		{
			ECS::TagComponent& tagComponent = entity.getComponent<ECS::TagComponent>();
			char buffer[256];
			memset(buffer, 0, sizeof(buffer));
			strcpy_s(buffer, tagComponent.Tag.c_str());
			if (ImGui::InputText(" ", buffer, sizeof(buffer)))
			{
				tagComponent.Tag = std::string(buffer);
			}
		}
		// Draw Mesh Component
		drawComponent<GFX::Mesh>(entity, "Mesh Filter", [](auto& component)
			{
				//SIByLEditor::DrawTriangleMeshSocket("Mesh", component);
			});

		//Component::onDrawGui((void*)(typeid(GFX::RDG::ResourceNode).hash_code() + 1), "Consume History", []() {


		{
			ImGui::Separator();
			ImGui::Dummy(ImVec2(0.0f, 20.0f));
			ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
			ImVec2 buttonSize(200, 30);

			ImGui::SetCursorPosX(contentRegionAvailable.x / 2 - 100);
			if (ImGui::Button("Add Component", buttonSize))
				ImGui::OpenPopup("AddComponent");

			if (ImGui::BeginPopup("AddComponent"))
			{
				if (ImGui::MenuItem("Mesh Filter"))
				{
					//entity.addComponent<GFX::Mesh>();
					ImGui::CloseCurrentPopup();
				}
				ImGui::EndPopup();
			}
		}
	}
}