module;
#include <vector>
#include <string>
#include <unordered_map>
#include "entt/entt.hpp"
export module GFX.SceneTree;
import ECS.Entity;
import ECS.UID;

namespace SIByL::GFX
{
	export struct SceneNode
	{
		ECS::Entity entity;
		uint64_t uid;
		uint64_t parent;
		std::vector<uint64_t> children;
	};

	export using SceneNodeHandle = uint64_t;

	export struct SceneTree
	{
		SceneTree();
		ECS::Context context;
		SceneNodeHandle root;
		std::unordered_map<SceneNodeHandle, SceneNode> nodes;

		auto getRootHandle() noexcept -> SceneNodeHandle { return root; }
		auto getNodeEntity(SceneNodeHandle const& handle) noexcept -> ECS::Entity& { return nodes[handle].entity; }

		auto addNode(std::string const& name, uint64_t const& parent) noexcept -> uint64_t;
		auto moveNode(uint64_t const& node, uint64_t const& parent) noexcept -> void;
		auto removeNode(uint64_t const&) noexcept -> void;

		auto print2Console() noexcept -> void;
		auto printNode2Console(SceneNodeHandle const& node, size_t bracket_num) noexcept -> void;
	};
}