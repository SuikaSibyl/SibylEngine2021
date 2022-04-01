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