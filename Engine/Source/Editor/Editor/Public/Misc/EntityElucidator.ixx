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
#include <glm/glm.hpp>
export module Editor.EntityElucidator;
import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;
import GFX.Mesh;
import GFX.Transform;
import GFX.Camera;
import Editor.CommonProperties;

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
				ImGui::Dummy(ImVec2(0.0f, 20.0f));
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
		drawComponent<GFX::Transform>(entity, "Transform", [](auto& component)
			{
				// set translation
				glm::vec3 translation = component.getTranslation();
				drawVec3Control("Translation", translation);
				bool translation_modified = (component.getTranslation() != translation);
				if (translation_modified) component.setTranslation(translation);
				// set rotation
				glm::vec3 rotation = component.getEulerAngles();
				drawVec3Control("Rotation", rotation);
				bool rotation_modified = (component.getEulerAngles() != rotation);
				if (rotation_modified) component.setEulerAngles(rotation);
				// set scale
				glm::vec3 scaling = component.getScale();
				drawVec3Control("Scaling", scaling, 1);
				bool scale_modified = (component.getScale() != scaling);
				if (scale_modified) component.setScale(scaling);
			});
		// Draw Camera Component
		drawComponent<GFX::Camera>(entity, "Camera", [](auto& component)
			{
				// fov
				static float radians2degree = 180.f/3.1415926f;
				float fov = radians2degree * component.getFovy();
				drawFloatControl("FoV", fov, 1, 1, 180, 45);
				bool fov_modified = (radians2degree*(component.getFovy()) != fov);
				if (fov_modified && fov > 0) component.setFovy(glm::radians(fov));
				// near
				float near = component.getNear();
				drawFloatControl("Near", near, 0.01, 0.001, 2, 0.1);
				bool near_modified = (component.getNear() != near);
				if (near_modified && near > 0) component.setNear(near);
				// near
				float far = component.getFar();
				drawFloatControl("Far", far, 1, 1, 1000, 100);
				bool far_modified = (component.getFar() != far);
				if (far_modified && far > 0) component.setFar(far);

				//SIByLEditor::DrawTriangleMeshSocket("Mesh", component);
			});
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
				if (ImGui::MenuItem("Transform"))
				{
					if (!entity.hasComponent<GFX::Transform>())
						entity.addComponent<GFX::Transform>();
					ImGui::CloseCurrentPopup();
				}
				if (ImGui::MenuItem("Camera"))
				{
					if (!entity.hasComponent<GFX::Camera>())
						entity.addComponent<GFX::Camera>();
					ImGui::CloseCurrentPopup();
				}
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