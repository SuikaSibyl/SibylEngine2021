module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <imgui_internal.h>
#include "entt.hpp"
export module Editor.Scene;
import Editor.Widget;
import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;
import GFX.SceneTree;
import GFX.Scene;
import Editor.Inspector;
import Editor.EntityElucidator;

namespace SIByL::Editor
{
	export struct Scene :public Widget
	{
		virtual auto onDrawGui() noexcept -> void override;

		auto bindScene(GFX::Scene* scene) { binded_scene = scene; }
		auto bindInspector(Inspector* inspector) noexcept -> void;
		auto drawNode(GFX::SceneNodeHandle const& node) -> void;
		GFX::Scene* binded_scene;

		GFX::SceneNodeHandle forceNodeOpen = 0;
		GFX::SceneNodeHandle inspected = 0;
		Inspector* inspector = nullptr;
	};

	auto Scene::onDrawGui() noexcept -> void
	{
		ImGui::Begin("Scene", 0, ImGuiWindowFlags_MenuBar);

		// Menu
		{
			ImGui::PushItemWidth(ImGui::GetFontSize() * -12);
			if (ImGui::BeginMenuBar())
			{
				// Menu - File
				if (ImGui::BeginMenu("File"))
				{
					// Menu - File - Load
					if (ImGui::MenuItem("Load"))
					{

					}
					// Menu - File - Save
					if (ImGui::MenuItem("Save"))
					{

					}
					// Menu - File - Save as
					if (ImGui::MenuItem("Save as"))
					{

					}
					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}
			ImGui::PopItemWidth();
		}

		// Left-clock on blank space
		if (ImGui::IsMouseDown(0) && ImGui::IsWindowHovered())
		{
			if (inspector) inspector->setCustomDraw(nullptr);
		}
		// Right-click on blank space
		if (ImGui::BeginPopupContextWindow(0, 1, false))
		{
			if (ImGui::MenuItem("Create Empty Entity"))
			{
				binded_scene->tree.addNode("Empty Node", binded_scene->tree.root);
				ImGui::SetNextItemOpen(true, ImGuiCond_Always);
			}
			ImGui::EndPopup();
		}


		if (binded_scene)
		{
			GFX::SceneTree& tree = binded_scene->tree;
			drawNode(tree.root);
		}

		ImGui::End();
	}

	auto Scene::drawNode(GFX::SceneNodeHandle const& node) -> void
	{
		GFX::SceneTree& tree = binded_scene->tree;
		ECS::TagComponent& tag = tree.nodes[node].entity.getComponent<ECS::TagComponent>();

		ImGuiTreeNodeFlags node_flags = 0;
		if (tree.nodes[node].children.size() == 0) node_flags |= ImGuiTreeNodeFlags_Leaf;

		if (node == forceNodeOpen)
		{
			ImGui::SetNextItemOpen(true, ImGuiCond_Always);
			forceNodeOpen = 0;
		}
		bool opened = ImGui::TreeNodeEx(tag.Tag.c_str(), node_flags);
		ImGuiID uid = ImGui::GetID(tag.Tag.c_str());
		ImGui::TreeNodeBehaviorIsOpen(uid);

		// Clicked
		if (ImGui::IsItemClicked())
		{
			ECS::Entity entity = binded_scene->tree.getNodeEntity(node);
			inspected = node;
			if (inspector) inspector->setCustomDraw(std::bind(EntityElucidator::drawInspector, entity));
		}

		// Right-click on blank space
		bool entityDeleted = false;
		if (ImGui::BeginPopupContextItem())
		{
			if (ImGui::MenuItem("Create Empty Entity"))
			{
				binded_scene->tree.addNode("Empty Node", node);
				forceNodeOpen = node;
			}
			if (node != binded_scene->tree.root)
			{
				if (ImGui::MenuItem("Delete Entity"))
					entityDeleted = true;
			}

			ImGui::EndPopup();
		}

		// If draged
		if (ImGui::BeginDragDropSource())
		{
			ImGui::Text(tag.Tag.c_str());
			ImGui::SetDragDropPayload("SceneEntity", &node, sizeof(node));
			ImGui::EndDragDropSource();
		}
		// If dragged to
		if (ImGui::BeginDragDropTarget())
		{
			if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SceneEntity"))
			{
				GFX::SceneNodeHandle* dragged_handle = (uint64_t*)payload->Data;
				binded_scene->tree.moveNode(*dragged_handle, node);
			}
			ImGui::EndDragDropTarget();
		}

		// Opened
		if (opened)
		{
			ImGui::NextColumn();
			for (int i = 0; i < tree.nodes[node].children.size(); i++)
			{
				ImGui::PushID(i);
				drawNode(tree.nodes[node].children[i]);
				ImGui::PopID();
			}
			ImGui::TreePop();
		}

		if (entityDeleted)
		{
			bool isParentOfInspected = false;
			GFX::SceneNodeHandle inspected_parent = inspected;
			while (inspected_parent != 0)
			{
				inspected_parent = binded_scene->tree.nodes[inspected_parent].parent;
				if (node == node) { isParentOfInspected = true; break; } // If set ancestor as child, no movement;
			}
			if (node == inspected || isParentOfInspected)
			{
				if (inspector) inspector->setCustomDraw(nullptr);
				inspected = 0;
			}
			binded_scene->tree.removeNode(node);
		}
	}

	auto Scene::bindInspector(Inspector* inspector) noexcept -> void
	{
		this->inspector = inspector;
	}
}