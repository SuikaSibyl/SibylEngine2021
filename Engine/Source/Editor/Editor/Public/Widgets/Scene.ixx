module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include "entt.hpp"
export module Editor.Scene;
import Editor.Widget;
import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;
import GFX.SceneTree;
import GFX.Scene;

namespace SIByL::Editor
{
	export struct Scene :public Widget
	{
		virtual auto onDrawGui() noexcept -> void override;

		auto bindScene(GFX::Scene* scene) { binded_scene = scene; }
		auto drawNode(GFX::SceneNodeHandle const& node) -> void;
		GFX::Scene* binded_scene;
	};

	auto Scene::onDrawGui() noexcept -> void
	{
		ImGui::Begin("Scene", 0, ImGuiWindowFlags_MenuBar);

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
		if (ImGui::TreeNode(tag.Tag.c_str()))
		{
			ImGui::NextColumn();
			for (int i = 0; i < tree.nodes[node].children.size(); i++)
			{
				drawNode(tree.nodes[node].children[i]);
			}
			ImGui::TreePop();
		}
	}
}