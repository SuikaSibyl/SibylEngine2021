#include "SIByLpch.h"
#include "DrawProperty.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/Core/Geometry/MeshLoader.h"
#include "Sibyl/ECS/Asset/AssetUtility.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"
#include "EditorLayer.h"

#include "Sibyl/ECS/Asset/AssetUtility.h"
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

		std::string Textname = GetShort(item.TextureID);
		ImGui::Button(Textname.c_str(), ImVec2(100.0f, 0.0f));
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ASSET"))
			{
				const wchar_t* path = (const wchar_t*)payload->Data;
				std::filesystem::path texturePath = std::filesystem::path("../Assets/") / path;
				SIByL::Ref<SIByL::Texture2D> texture = SIByL::Library<SIByL::Texture2D>::Fetch(PathToIdentifier(texturePath.string()));
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
		
		std::string savePath = GetShort(mesh.Mesh->m_Path);

		ImGui::Button(savePath.c_str(), ImVec2(100.0f, 0.0f));
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

	void DrawMeshRendererMaterialSocket(const std::string& label, SIByL::MeshRendererComponent& meshRenderer, int i)
	{
		ImGui::Text(label.c_str());
		ImGui::SameLine();

		ImGui::PushID(label.c_str());

		std::string savePath = GetShort(meshRenderer.Materials[i]->GetSavePath());

		ImGui::Button(savePath.c_str(), ImVec2(100.0f, 0.0f));
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ASSET"))
			{
				const wchar_t* path = (const wchar_t*)payload->Data;
				std::filesystem::path materialPath = path;
				SIByL::Ref<SIByL::Material> mat = SIByL::GetAssetByPath<SIByL::Material>(materialPath.string());
				meshRenderer.Materials[i] = mat;
			}
			ImGui::EndDragDropTarget();
		}

		ImGui::PopID();
	}

	void DrawShaderSlot(const std::string& label, SIByL::Material& material)
	{		
		std::string ShaderName = "Fallback";
		Ref<Shader> shaderUsed = material.GetShaderUsed();
		if (shaderUsed != nullptr)
			ShaderName = shaderUsed->ShaderID;

		ImGui::Text("Shader: ");
		ImGui::SameLine();

		if (ImGui::Button(ShaderName.c_str()))
			ImGui::OpenPopup("Shaders Library");

		if (ImGui::BeginPopup("Shaders Library"))
		{
			static int selected = -1;
			int index = 0;
			for (auto& iter : SIByL::Library<SIByL::Shader>::Mapper)
			{
				if (ImGui::Selectable(iter.first.c_str(), selected == index))
				{
					selected = index;
					material.UseShader(iter.second);
					ImGui::CloseCurrentPopup();
				}
				index++;
			}

			ImGui::EndPopup();
		}
	}

	void DrawMaterialSlot(const std::string& label)
	{

	}

	void DrawMaterial(const std::string& label, SIByL::Material& material)
	{
		const ImGuiTreeNodeFlags treeNodeFlags =
			ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_FramePadding |
			ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap;

		std::string title = "Material : " + GetShort(material.SavePath);
		if (material.IsAssetDirty)
		{
			title += "*";
		}

		ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4,4 });
		float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		ImGui::Separator();
		bool opened = ImGui::TreeNodeEx((void*)1000, treeNodeFlags, title.c_str());
		ImGui::PopStyleVar();
		if (material.IsAssetDirty)
		{
			ImGui::SameLine(contentRegionAvailable.x - lineHeight * 1);
			if (ImGui::Button("Save"))
				material.SaveAsset();
		}

		if (opened)
		{
			// If Material Already Exist
			if (&material != nullptr)
			{
				DrawShaderSlot("Shader", material);

				// ================================================================
				// Draw constants
				SIByL::ShaderConstantsDesc* constant = material.GetConstantsDesc();

				if (constant != nullptr)
				{
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
								material.SetAssetDirty();
							}
							ImGui::PopID();
						}
					}
				}

				// ================================================================
				// Draw resources
				SIByL::ShaderResourcesDesc* resources = material.GetResourcesDesc();

				if (resources != nullptr)
				{
					for (auto iter : *resources)
					{
						SIByL::ShaderResourceItem& item = iter.second;

						DrawTexture2D(material, item);
					}
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