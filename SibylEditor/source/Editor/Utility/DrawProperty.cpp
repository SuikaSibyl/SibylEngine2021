#include "SIByLpch.h"
#include "DrawProperty.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"
#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/Core/Geometry/MeshLoader.h"

#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"

namespace SIByLEditor
{
	void DrawVec3Control(const std::string& label, glm::vec3& values, float resetValue, float columeWidth)
	{
		ImGuiIO& io = ImGui::GetIO();
		auto boldFont = io.Fonts->Fonts[0];

		ImGui::PushID(label.c_str());

		ImGui::Columns(2);
		ImGui::SetColumnWidth(0, columeWidth);
		ImGui::Text(label.c_str());
		ImGui::NextColumn();

		ImGui::PushMultiItemsWidths(3, ImGui::CalcItemWidth());
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2{ 0,0 });

		float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		ImVec2 buttonSize = { lineHeight + 3.0f, lineHeight };

		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.8,0.1f,0.15f,1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.9,0.2f,0.2f,1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.8,0.1f,0.15f,1.0f });
		ImGui::PushFont(boldFont);
		if (ImGui::Button("X", buttonSize))
			values.x = resetValue;
		ImGui::PopFont();
		ImGui::PopStyleColor(3);
		ImGui::SameLine();
		ImGui::DragFloat("##x", &values.x, 0.1f);
		ImGui::PopItemWidth();
		ImGui::SameLine();

		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.2,0.7f,0.2f,1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.3,0.8f,0.3f,1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.2,0.7f,0.2f,1.0f });
		ImGui::PushFont(boldFont);
		if (ImGui::Button("Y", buttonSize))
			values.y = resetValue;
		ImGui::PopFont();
		ImGui::PopStyleColor(3);
		ImGui::SameLine();
		ImGui::DragFloat("##y", &values.y, 0.1f);
		ImGui::PopItemWidth();
		ImGui::SameLine();

		ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.1,0.26f,0.8f,1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4{ 0.2,0.35f,0.9f,1.0f });
		ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4{ 0.1,0.26f,0.8f,1.0f });
		ImGui::PushFont(boldFont);
		if (ImGui::Button("Z", buttonSize))
			values.z = resetValue;
		ImGui::PopFont();
		ImGui::PopStyleColor(3);
		ImGui::SameLine();
		ImGui::DragFloat("##z", &values.z, 0.1f);
		ImGui::PopItemWidth();
		ImGui::SameLine();

		ImGui::PopStyleVar();
		ImGui::Columns(1);

		ImGui::PopID();
	}

	void DrawTexture2D(SIByL::Material& material, SIByL::ShaderResourceItem& item)
	{
		ImGui::PushID(item.Name.c_str());
		ImGui::Text(item.Name.c_str());
		ImGui::SameLine();

		ImGui::Button("Texture", ImVec2(100.0f, 0.0f));
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ASSET"))
			{
				const wchar_t* path = (const wchar_t*)payload->Data;
				std::filesystem::path texturePath = std::filesystem::path("../Assets/") / path;
				SIByL::Ref<SIByL::Texture2D> texture = SIByL::Texture2D::Create(texturePath.string());
				material.SetTexture2D(item.Name, texture);
			}
			ImGui::EndDragDropTarget();
		}
		ImGui::PopID();
	}

	void DrawTriangleMeshSocket(const std::string& label, SIByL::MeshFilterComponent& mesh)
	{
		static const std::string name = "Mesh";
		ImGui::PushID(label.c_str());
		ImGui::Text(name.c_str());
		ImGui::SameLine();

		ImGui::Button("Mesh", ImVec2(100.0f, 0.0f));
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ASSET"))
			{
				SIByL::VertexBufferLayout layout =
				{
					{SIByL::ShaderDataType::Float3, "POSITION"},
					{SIByL::ShaderDataType::Float2, "TEXCOORD"},
				};

				const wchar_t* path = (const wchar_t*)payload->Data;
				std::filesystem::path texturePath = path;
				SIByL::MeshLoader meshLoader(texturePath.string(), layout);
				mesh.Mesh = meshLoader.GetTriangleMesh();
			}
			ImGui::EndDragDropTarget();
		}
		ImGui::PopID();
	}

	void DrawMaterial(const std::string& label, SIByL::Material& material)
	{
		ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth;
		bool opened = ImGui::TreeNodeEx((void*)1000, flags, "Material");
		if (opened)
		{
			// If Material Already Exist
			if (&material != nullptr)
			{
				// ================================================================
				// Draw constants
				SIByL::ShaderConstantsDesc* constant = material.GetConstantsDesc();

				for (auto iter : *constant)
				{
					SIByL::ShaderConstantItem& item = iter.second;

					ImGui::Text(item.Name.c_str());
					ImGui::SameLine();

					if (item.Type == SIByL::ShaderDataType::RGB)
					{

					}
					else if (item.Type == SIByL::ShaderDataType::RGBA)
					{
						ImGui::PushID(item.Name.c_str());
						float* color = material.PtrFloat4(item.Name);
						if (ImGui::ColorEdit4(" ", color))
						{
							material.SetDirty();
						}
						ImGui::PopID();
					}
				}

				// ================================================================
				// Draw resources
				SIByL::ShaderResourcesDesc* resources = material.GetResourcesDesc();

				for (auto iter : *resources)
				{
					SIByL::ShaderResourceItem& item = iter.second;

					DrawTexture2D(material, item);
				}
			}
			// If Material Not Exist Yet
			else
			{

			}

			ImGui::TreePop();
		}
	}
}