module;
#include <vector>
#include <string>
#include "entt/entt.hpp"
export module GFX.SceneTree;
import ECS.Entity;
import ECS.UID;

namespace SIByL::GFX
{
	export struct SceneNode
	{
		ECS::Entity entity;
		SceneNode* parent;
		std::vector<SceneNode*> children;
	};

	export struct SceneTree
	{
		SceneTree();
		ECS::Context context;
		std::vector<SceneNode> nodes;

		auto addNode(std::string const& name, SceneNode* parent) noexcept -> SceneNode*;
		auto moveNode(SceneNode* node, SceneNode* parent) noexcept -> void;
		auto removeNode(SceneNode*) noexcept -> void;

		auto print2Console() noexcept -> void;
		auto printNode2Console(SceneNode* node, size_t bracket_num) noexcept -> void;
	};
}