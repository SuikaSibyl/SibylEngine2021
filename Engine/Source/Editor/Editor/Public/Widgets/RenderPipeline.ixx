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
				for (int i = 0; i < rg->resources.size(); i++)
				{
					bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)i, bullet_flags, std::string(rg->getResourceNode(rg->resources[i])->tag).c_str(), i);
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
			Component::onDrawGui((void*)(typeid(GFX::RDG::ResourceNode).hash_code() + 0), "Resource Information", []() {

				}, []() {});

			Component::onDrawGui((void*)(typeid(GFX::RDG::ResourceNode).hash_code() + 1), "Consume History", []() {

				}, []() {});
		}

	}
}