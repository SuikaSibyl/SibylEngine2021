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
#include <filesystem>
#include <glm/glm.hpp>
export module Editor.EntityElucidator;
import Asset.Asset;
import Asset.Mesh;
import Asset.AssetLayer;
import Asset.MeshLoader;
import Asset.RuntimeAssetManager;
import Asset.DedicatedLoader;

import Core.Log;
import Core.Hash;
import Core.Buffer;
import Core.Cache;

import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;
import GFX.Mesh;
import GFX.Transform;
import GFX.Camera;
import GFX.Renderer;
import GFX.SceneTree;
import GFX.Scene;
import Editor.CommonProperties;

import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.RasterPassNode;
import GFX.RDG.Common;
import GFX.RDG.MultiDispatchScope;
import GFX.Renderer;
import GFX.RDG.RasterNodes;
import GFX.RDG.ExternalAccess;

namespace SIByL::Editor
{
	export struct EntityElucidator
	{
		static auto drawInspector(ECS::Entity entity, Asset::AssetLayer* assetLayer, GFX::SceneNodeHandle const& node, GFX::Scene* scene, GFX::RDG::RenderGraph* renderGraph) -> void;
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

	auto snifferExternalMeshForASingleNode(Asset::ExternalMeshSniffer::Node node, Asset::ExternalMeshSniffer* sniffer, GFX::SceneNodeHandle const& parent, GFX::Scene* scene, Asset::AssetLayer* assetLayer) noexcept -> void
	{
		uint32_t mesh_nums, children_nums;
		std::string name;
		sniffer->interpretNode(node, mesh_nums, children_nums, name);
		GFX::SceneNodeHandle new_parent = scene->tree.addNode(name, parent);
		for (uint32_t i = 0; i < children_nums; i++)
		{
			auto child = sniffer->getNodeChildren(node, i);
			snifferExternalMeshForASingleNode(child, sniffer, new_parent, scene, assetLayer);
		}
		if (mesh_nums > 0)
		{
			auto entity = scene->tree.nodes[new_parent].entity;
			auto& mesh_component = entity.addComponent<GFX::Mesh>();
			Asset::Mesh tmp_mesh;
			Asset::MeshLoader tmp_loader(tmp_mesh);
			uint64_t cache_id = ECS::UniqueID::RequestUniqueID();
			sniffer->fillVertexIndex(node, tmp_loader.vb, tmp_loader.ib);
			tmp_loader.saveAsCache(cache_id);
			Asset::ResourceItem new_item = 
			{
				Asset::ResourceKind::MESH,
				0,
				cache_id,
			};
			uint64_t guid = assetLayer->runtimeManager.addNewAsset(new_item);
			mesh_component = GFX::Mesh::query(guid, assetLayer);
		}
	}

	auto EntityElucidator::drawInspector(ECS::Entity entity, Asset::AssetLayer* assetLayer, GFX::SceneNodeHandle const& node, GFX::Scene* scene, GFX::RDG::RenderGraph* renderGraph) -> void
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
			});
		// Draw Mesh Component
		drawComponent<GFX::Mesh>(entity, "Mesh Filter", [assetLayer = assetLayer, node = node, scene = scene](auto& component)
			{
				// Position
				{
					bool hasPosition = (component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::POSITION) != 0;
					ImGui::Checkbox("POSITION", &hasPosition);
					if (hasPosition != ((component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::POSITION) != 0))
						component.meshDesc.vertexInfo ^= (uint32_t)Asset::VertexInfoBits::POSITION;
				}
				ImGui::SameLine();
				// Color
				{
					bool hasColor = (component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::COLOR) != 0;
					ImGui::Checkbox("COLOR", &hasColor);
					if (hasColor != ((component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::COLOR) != 0))
						component.meshDesc.vertexInfo ^= (uint32_t)Asset::VertexInfoBits::COLOR;
				}
				ImGui::SameLine();
				// UV
				{
					bool hasUV = (component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::UV) != 0;
					ImGui::Checkbox("UV", &hasUV);
					if (hasUV != ((component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::UV) != 0))
						component.meshDesc.vertexInfo ^= (uint32_t)Asset::VertexInfoBits::UV;
				}
				// NORMAL
				{
					bool hasNormal = (component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::NORMAL) != 0;
					ImGui::Checkbox("NORMAL ", &hasNormal);
					if (hasNormal != ((component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::NORMAL) != 0))
						component.meshDesc.vertexInfo ^= (uint32_t)Asset::VertexInfoBits::NORMAL;
				}
				ImGui::SameLine();
				// TANGENT
				{
					bool hasTangent = (component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::TANGENT) != 0;
					ImGui::Checkbox("TANGENT", &hasTangent);
					if (hasTangent != ((component.meshDesc.vertexInfo & (uint32_t)Asset::VertexInfoBits::TANGENT) != 0))
						component.meshDesc.vertexInfo ^= (uint32_t)Asset::VertexInfoBits::TANGENT;
				}
				// Socket
				{
					std::string guid = std::to_string(component.guid);
					ImGui::PushID(guid.c_str());
					ImGui::Text("Mesh GUID");
					ImGui::SameLine();

					ImGui::Button(guid.c_str(), ImVec2(250.0f, 0.0f));
					if (ImGui::BeginDragDropTarget())
					{
						if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("ASSET"))
						{
							const wchar_t* path = (const wchar_t*)payload->Data;
							std::filesystem::path meshPath = path;
							SE_CORE_INFO("Dragged Path: {0}", meshPath);
							Asset::GUID guid = Hash::path2hash(meshPath);
							auto find_result = assetLayer->runtimeManager.findAsset(guid);
							if (!find_result)//if not already exists
							{
								Asset::ExternalMeshSniffer sniffer;
								uint32_t mesh_nums, children_nums;
								std::string name;
								auto parent_node = sniffer.loadFromFile(meshPath);
								sniffer.interpretNode(parent_node, mesh_nums, children_nums, name);
								for (uint32_t i = 0; i < children_nums; i++)
								{
									auto child = sniffer.getNodeChildren(parent_node, i);
									snifferExternalMeshForASingleNode(child, &sniffer, node, scene, assetLayer);
								}
							}
							else
							{
								component = GFX::Mesh::query(guid, assetLayer);
							}
						}
						ImGui::EndDragDropTarget();
					}
					ImGui::PopID();
				}
			});
		// Draw Renderer Component
		drawComponent<GFX::Renderer>(entity, "Renderer", [&renderGraph = renderGraph ](auto& component)
			{
				static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;

				int remov_subRenderer = -1;
				for (int i = 0; i < component.subRenderers.size(); i++)
				{
					auto& subRenderer = component.subRenderers[i];
					ImGui::PushID(i);

					if (ImGui::BeginTable("SubPass", 2, flags))
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("Pass Name");
						ImGui::TableSetColumnIndex(1);
						ImGui::Text(subRenderer.passName.c_str());
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("Pipeline Name");
						ImGui::TableSetColumnIndex(1);
						ImGui::Text(subRenderer.pipelineName.c_str());
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("Material Name");
						ImGui::TableSetColumnIndex(1);
						ImGui::Text(subRenderer.materialName.c_str());
						ImGui::EndTable();
					}
					ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

					ImGui::SetCursorPosX(contentRegionAvailable.x - 200);
					ImVec2 buttonSize(100, 30);
					if (ImGui::Button("Choose", buttonSize))
						ImGui::OpenPopup("choose_material");
					if (ImGui::BeginPopup("choose_material"))
					{
						for (auto iter : renderGraph->rasterPassRegister)
						{
							auto pass_scope = renderGraph->getNode<GFX::RDG::RasterPassScope>(iter.second);
							if (ImGui::BeginMenu(iter.first.c_str()))
							{
								std::string pass_name = iter.first;
								for (auto iter : pass_scope->pipelineScopesRegister)
								{
									auto pipeline_scope = renderGraph->getNode<GFX::RDG::RasterPipelineScope>(iter.second);
									if (ImGui::BeginMenu(iter.first.c_str()))
									{
										std::string pipeline_name = iter.first;
										auto material_scope = renderGraph->getNode<GFX::RDG::RasterMaterialScope>(iter.second);
										for (auto iter : pipeline_scope->materialScopesRegister)
										{
											std::string material_name = iter.first;
											bool is_choosen = false;
											ImGui::MenuItem(iter.first.c_str(), "", &is_choosen);
											if (is_choosen)
											{
												subRenderer.passName = pass_name;
												subRenderer.pipelineName = pipeline_name;
												subRenderer.materialName = material_name;
											}
										}
										ImGui::EndMenu();
									}
								}
								ImGui::EndMenu();
							}
						}
						//for (int i = 0; i < IM_ARRAYSIZE(names); i++)
						//	ImGui::MenuItem(names[i], "", &toggles[i]);

						//ImGui::Separator();
						//ImGui::Text("Tooltip here");
						//if (ImGui::IsItemHovered())
						//	ImGui::SetTooltip("I am a tooltip over a popup");

						//if (ImGui::Button("Stacked Popup"))
						//	ImGui::OpenPopup("another popup");
						//if (ImGui::BeginPopup("another popup"))
						//{
						//	for (int i = 0; i < IM_ARRAYSIZE(names); i++)
						//		ImGui::MenuItem(names[i], "", &toggles[i]);
						//	if (ImGui::BeginMenu("Sub-menu"))
						//	{
						//		ImGui::MenuItem("Click me");
						//		if (ImGui::Button("Stacked Popup"))
						//			ImGui::OpenPopup("another popup");
						//		if (ImGui::BeginPopup("another popup"))
						//		{
						//			ImGui::Text("I am the last one here.");
						//			ImGui::EndPopup();
						//		}
						//		ImGui::EndMenu();
						//	}
						//	ImGui::EndPopup();
						//}
						ImGui::EndPopup();
					}

					ImGui::SameLine(contentRegionAvailable.x - 80);
					if (ImGui::Button("Remove", buttonSize))
						remov_subRenderer = i;

					ImGui::PopID();
					ImGui::Separator();
				}

				if (remov_subRenderer >= 0)
					component.subRenderers.erase(component.subRenderers.begin() + remov_subRenderer);

				// Draw Add SubPass
				ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();
				ImGui::SetCursorPosX(contentRegionAvailable.x / 2 - 50);
				ImVec2 buttonSize(100, 30);
				if (ImGui::Button("Add Pass", buttonSize))
					component.subRenderers.emplace_back("NONE", "NONE", "NONE");
			});

		// Add Components
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
					if (!entity.hasComponent<GFX::Mesh>())
						entity.addComponent<GFX::Mesh>();
					ImGui::CloseCurrentPopup();
				}
				if (ImGui::MenuItem("Renderer"))
				{
					if (!entity.hasComponent<GFX::Renderer>())
						entity.addComponent<GFX::Renderer>();
					ImGui::CloseCurrentPopup();
				}
				ImGui::EndPopup();
			}
		}
	}
}