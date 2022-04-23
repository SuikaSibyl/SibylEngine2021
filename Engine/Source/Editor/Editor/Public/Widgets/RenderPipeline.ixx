module;
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <imgui_internal.h>
#include "entt.hpp"
export module Editor.RenderPipeline;
import Core.String;
import Editor.Widget;
import Editor.Scene;
import Editor.Inspector;
import Editor.Component;
import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;
import GFX.SceneTree;
import GFX.Scene;
import GFX.RDG.RenderGraph;
import GFX.RDG.StorageBufferNode;
import GFX.RDG.RasterPassNode;
import GFX.RDG.Common;
import GFX.RDG.MultiDispatchScope;
import GFX.RDG.RasterNodes;
import GFX.RDG.ComputeSeries;
import GFX.RDG.ColorBufferNode;
import GFX.RDG.TextureBufferNode;

namespace SIByL::Editor
{
	export struct RenderPipeline :public Widget
	{
		virtual auto onDrawGui() noexcept -> void override;

		auto bindRenderGraph(GFX::RDG::RenderGraph* rg) noexcept -> void { this->rg = rg; }
		auto bindInspector(Inspector* ins) noexcept -> void { inspector = ins; }

		auto onDrawInspector() -> void;
		auto kickInspector() noexcept -> void;

		enum struct SelectedType
		{
			NONE,
			RESOURCE,
		};
		SelectedType selectedType = SelectedType::NONE;
		uint32_t selectedId;

		GFX::RDG::RenderGraph* rg;
		Inspector* inspector;
	};

	auto RenderPipeline::onDrawGui() noexcept -> void
	{
		ImGui::Begin("Render Pipeline", 0, ImGuiWindowFlags_MenuBar);

		static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;
		static ImGuiTreeNodeFlags bullet_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen; // ImGuiTreeNodeFlags_Bullet
		static bool test_drag_and_drop = false;

		if (rg)
		{
			if (ImGui::TreeNode("Resources Registered"))
			{
				ImGui::NextColumn();

				static bool show_COLOR_TEXTURE = true;
				static bool show_DEPTH_TEXTURE = true;
				static bool show_STORAGE_BUFFER = true;
				static bool show_UNIFORM_BUFFER = true;
				static bool show_FRAME_BUFFER = true;
				ImGui::Checkbox("COLOR_TEXTURE", &show_COLOR_TEXTURE);
				ImGui::Checkbox("DEPTH_TEXTURE", &show_DEPTH_TEXTURE);
				ImGui::Checkbox("STORAGE_BUFFER", &show_STORAGE_BUFFER);
				ImGui::Checkbox("UNIFORM_BUFFER", &show_UNIFORM_BUFFER);
				ImGui::Checkbox("FRAME_BUFFER", &show_FRAME_BUFFER);


				for (int i = 0; i < rg->resources.size(); i++)
				{
					auto resourceNode = rg->getResourceNode(rg->resources[i]);
					if (resourceNode->type == GFX::RDG::NodeDetailedType::COLOR_TEXTURE && !show_COLOR_TEXTURE) continue;
					if (resourceNode->type == GFX::RDG::NodeDetailedType::DEPTH_TEXTURE && !show_DEPTH_TEXTURE) continue;
					if (resourceNode->type == GFX::RDG::NodeDetailedType::STORAGE_BUFFER && !show_STORAGE_BUFFER) continue;
					if (resourceNode->type == GFX::RDG::NodeDetailedType::UNIFORM_BUFFER && !show_UNIFORM_BUFFER) continue;
					if (resourceNode->type == GFX::RDG::NodeDetailedType::FRAME_BUFFER && !show_FRAME_BUFFER) continue;

					bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, bullet_flags, std::string(resourceNode->tag).c_str(), i);
					if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
					{
						selectedType = SelectedType::RESOURCE;
						selectedId = i;
						kickInspector();
					}
					//if (test_drag_and_drop && ImGui::BeginDragDropSource())
					//{
					//	ImGui::SetDragDropPayload("_TREENODE", NULL, 0);
					//	ImGui::Text("This is a drag and drop source");
					//	ImGui::EndDragDropSource();
					//}
					//if (node_open)
					//{
					//	ImGui::BulletText("Blah blah\nBlah Blah");
					//	ImGui::TreePop();
					//}
				}
				ImGui::TreePop();
			}

			ImGui::PushID("Pass List");
			if (ImGui::TreeNode("Pass List"))
			{
				for (int i = 0; i < rg->passList.size(); i++)
				{
					ImGui::PushID(i);
					auto pass_node_handle = rg->passList[i];
					bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, base_flags, std::string(rg->getResourceNode(rg->passList[i])->tag).c_str(), i);
					//if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
					//{
					//	selectedType = SelectedType::RESOURCE;
					//	selectedId = i;
					//	kickInspector();
					//}
					if (node_open)
					{
						GFX::RDG::PassNode* passnode = (GFX::RDG::PassNode*)(rg->registry.getNode(pass_node_handle));
						// if pass_node is a RasterPassScope
						if (passnode->type == GFX::RDG::NodeDetailedType::RASTER_PASS_SCOPE)
						{
							GFX::RDG::RasterPassScope* pass_node = (GFX::RDG::RasterPassScope*)passnode;
							for (int i = 0; i < pass_node->pipelineScopes.size(); i++)
							{
								auto pipeline_node_handle = pass_node->pipelineScopes[i];
								GFX::RDG::RasterPipelineScope* pipeline_node = (GFX::RDG::RasterPipelineScope*)(rg->registry.getNode(pipeline_node_handle));
								ImGui::PushID(i);
								bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, base_flags, pipeline_node->tag.c_str(), i);
								if (node_open)
								{
									// all material passes
									for (int i = 0; i < pipeline_node->materialScopes.size(); i++)
									{
										auto material_node_handle = pipeline_node->materialScopes[i];
										GFX::RDG::RasterMaterialScope* material_node = (GFX::RDG::RasterMaterialScope*)(rg->registry.getNode(material_node_handle));
										ImGui::PushID(i);
										bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, base_flags, material_node->tag.c_str(), i);
										if (node_open)
										{
											// all draw call
											for (int i = 0; i < material_node->validDrawcallCount; i++)
											{
												auto drawcall_handle = material_node->drawCalls[i];
												GFX::RDG::RasterDrawCall* drawcall_node = (GFX::RDG::RasterDrawCall*)(rg->registry.getNode(drawcall_handle));
												bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, bullet_flags, drawcall_node->tag.c_str(), i);
												//if (node_open)
												//{
												//	ImGui::TreePop();
												//}
											}

											ImGui::TreePop();
										}
										ImGui::PopID();
									}
									ImGui::TreePop();
								}
								ImGui::PopID();
							}
						}
						// if pass_node is a ComputePassScope
						if (passnode->type == GFX::RDG::NodeDetailedType::COMPUTE_PASS_SCOPE)
						{
							GFX::RDG::ComputePassScope* pass_node = (GFX::RDG::ComputePassScope*)passnode;
							for (int i = 0; i < pass_node->pipelineScopes.size(); i++)
							{
								auto pipeline_node_handle = pass_node->pipelineScopes[i];
								GFX::RDG::ComputePipelineScope* pipeline_node = (GFX::RDG::ComputePipelineScope*)(rg->registry.getNode(pipeline_node_handle));
								ImGui::PushID(i);
								bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, base_flags, pipeline_node->tag.c_str(), i);
								if (node_open)
								{
									// all material passes
									for (int i = 0; i < pipeline_node->materialScopes.size(); i++)
									{
										auto material_node_handle = pipeline_node->materialScopes[i];
										GFX::RDG::ComputeMaterialScope* material_node = (GFX::RDG::ComputeMaterialScope*)(rg->registry.getNode(material_node_handle));
										ImGui::PushID(i);
										bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, base_flags, material_node->tag.c_str(), i);
										if (node_open)
										{
											// all draw call
											for (int i = 0; i < material_node->dispatches.size(); i++)
											{
												auto dispatch_handle = material_node->dispatches[i];
												GFX::RDG::ComputeDispatch* dispatch_node = (GFX::RDG::ComputeDispatch*)(rg->registry.getNode(dispatch_handle));
												bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, bullet_flags, dispatch_node->tag.c_str(), i);
											}

											ImGui::TreePop();
										}
										ImGui::PopID();
									}
									ImGui::TreePop();
								}
								ImGui::PopID();
							}
						}
						else if (passnode->type == GFX::RDG::NodeDetailedType::EXTERNAL_ACCESS_PASS)
						{

						}
						ImGui::TreePop();
					}
					ImGui::PopID();
				}
				ImGui::TreePop();
			}
			ImGui::PopID();
		}

		ImGui::End();
	}

	auto RenderPipeline::kickInspector() noexcept -> void
	{
		if (inspector) inspector->setCustomDraw(std::bind(&RenderPipeline::onDrawInspector, this));
	}

	auto RenderPipeline::onDrawInspector() -> void
	{
		if (selectedType == SelectedType::NONE)
		{
			return;
		}
		else if (selectedType == SelectedType::RESOURCE)
		{
			GFX::RDG::ResourceNode* resourceNode = rg->getResourceNode(rg->resources[selectedId]);
			char buffer[256];
			memset(buffer, 0, sizeof(buffer));
			strcpy_s(buffer, rg->getResourceNode(rg->resources[selectedId])->tag.c_str());
			if (ImGui::InputText(" ", buffer, sizeof(buffer)))
			{
				resourceNode->tag = std::string(buffer);
			}
			Component::onDrawGui((void*)(typeid(GFX::RDG::ResourceNode).hash_code() + 0), "Resource Information", [&]() {
				const ImGuiTreeNodeFlags treeNodeFlags =
					ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_FramePadding |
					ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap;
				static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;

				ImGui::NextColumn();
				if (ImGui::BeginTable(resourceNode->tag.c_str(), 2, flags))
				{
					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::Text("Type");
					ImGui::TableSetColumnIndex(1);
					switch (resourceNode->type)
					{
					case GFX::RDG::NodeDetailedType::NONE:
						ImGui::Text("None");
						break;
					case GFX::RDG::NodeDetailedType::SAMPLER:
						ImGui::Text("Sampler");
						break;
					case GFX::RDG::NodeDetailedType::COLOR_TEXTURE:
					{
						ImGui::Text("Color Texture");
						GFX::RDG::ColorBufferNode* colorBufferNode = (GFX::RDG::ColorBufferNode*)resourceNode;
						uint64_t native_handle = colorBufferNode->getTexture()->getNativeHandle();
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("Native Image");
						ImGui::TableSetColumnIndex(1);
						ImGui::Text(to_hex_string(native_handle).c_str());
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("First Layout");
						ImGui::TableSetColumnIndex(1);
						ImGui::Text(RHI::to_string(colorBufferNode->first_layout).c_str());
					}
						break;
					case GFX::RDG::NodeDetailedType::DEPTH_TEXTURE:
						ImGui::Text("Depth Texture");
						break;
					case GFX::RDG::NodeDetailedType::STORAGE_BUFFER:
						ImGui::Text("Storage Buffer");
						break;
					case GFX::RDG::NodeDetailedType::UNIFORM_BUFFER:
						ImGui::Text("Uniform Buffer");
						break;
					case GFX::RDG::NodeDetailedType::FRAME_BUFFER:
						ImGui::Text("Frame Buffer");
						break;
					case GFX::RDG::NodeDetailedType::FRAME_BUFFER_FLIGHTS:
						ImGui::Text("Frame Buffer Flights");
						break;
					case GFX::RDG::NodeDetailedType::UNIFORM_BUFFER_FLIGHTS:
						ImGui::Text("Uniform Buffer Flights");
						break;
					default:
						ImGui::Text("UNKNOWN");
						break;
					}
					ImGui::EndTable();
				}
				}, []() {});

			Component::onDrawGui((void*)(typeid(GFX::RDG::ResourceNode).hash_code() + 1), "Consume History", [&]() {
				static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
				ImGui::NextColumn();
				if (ImGui::BeginTable("Consume History", 2, flags))
				{
					for (auto& consumeHistory : resourceNode->consumeHistory)
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text(GFX::RDG::getString(consumeHistory.kind).c_str());
						ImGui::TableSetColumnIndex(1);
						ImGui::Text(GFX::RDG::getString(rg->getPassNode(consumeHistory.pass)->type).c_str());
					}
					ImGui::EndTable();
				}

				}, []() {});

			Component::onDrawGui((void*)(typeid(GFX::RDG::ResourceNode).hash_code() + 2), "Barriers", [&]() {
				static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;
				ImGui::NextColumn();
				if (ImGui::BeginTable("Barriers", 2, flags))
				{
					for (auto& pair : resourceNode->createdBarriers)
					{
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text(std::to_string(pair.first).c_str());
						ImGui::TableSetColumnIndex(1);
						ImGui::Text("whatever");
					}
					ImGui::EndTable();
				}

				}, []() {});
		}

	}
}