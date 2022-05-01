module;
#include <utility>
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <glm/glm.hpp>
#include <algorithm>
#include "entt/entt.hpp"
module GFX.SceneTree;
import Core.Log;
import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;
import GFX.Transform;

namespace SIByL::GFX
{
	SceneTree::SceneTree()
	{
		root = addNode("NewScene", 0);
	}

	auto SceneTree::appointRoot(SceneNodeHandle const& new_root) noexcept -> void
	{
		removeNode(root);
		root = new_root;
	}

	auto SceneTree::getNodeEntity(std::string const& name) noexcept -> ECS::Entity
	{
		for (auto node : nodes)
		{
			std::string& tag = node.second.entity.getComponent<ECS::TagComponent>().Tag;
			if (tag == name)
				return (node.second.entity);
		}
		return {};
	}

	auto SceneTree::addNode(std::string const& name, SceneNodeHandle const& parent) noexcept -> SceneNodeHandle
	{
		uint64_t uid = ECS::UniqueID::RequestUniqueID();
		addNode(name, uid, parent, {});

		// add children to parent
		if (parent != 0) // if not root
			nodes[parent].children.emplace_back(uid);

		return uid;
	}

	auto SceneTree::addNode(std::string const& name, uint64_t const& uid, uint64_t const& parent, std::vector<uint64_t>&& children) noexcept -> uint64_t
	{
		ECS::Entity entity = context.createEntity(name);
		entity.addComponent<GFX::Transform>();
		nodes[uid] = SceneNode{ entity, uid, parent, std::move(children)};

		return uid;
	}

	auto SceneTree::moveNode(uint64_t const& handle, uint64_t const& parent) noexcept -> void
	{
		uint64_t handle_to_move = handle;
		// check whether parent is not decestors
		uint64_t parent_cursor = parent;
		while (parent_cursor != 0)
		{
			parent_cursor = nodes[parent_cursor].parent;
			if (parent_cursor == handle) return; // If set ancestor as child, no movement;
		}
		// remove node from its previous parent
		// remove from parent
		auto& parent_children = nodes[nodes[handle_to_move].parent].children;
		for (int i = 0; i < parent_children.size(); i++)
		{
			if (parent_children[i] == handle_to_move)
			{
				parent_children.erase(parent_children.begin() + i);
				break;
			}
		}
		// add node to its new parent
		nodes[parent].children.emplace_back(handle_to_move);
		// change the node info
		nodes[handle].parent = parent;
	}

	auto SceneTree::removeNode(uint64_t const& handle) noexcept -> void
	{
		uint64_t handle_to_delete = handle;
		context.destroyEntity(nodes[handle_to_delete].entity);
		for (int i = 0; i < nodes[handle_to_delete].children.size(); i++)
		{
			removeNode(nodes[handle_to_delete].children[i]);
		}
		// remove from parent
		if (nodes[handle].parent != 0)
		{
			auto& parent_children = nodes[nodes[handle_to_delete].parent].children;
			for (int i = 0; i < parent_children.size(); i++)
			{
				if (parent_children[i] == handle_to_delete)
				{
					parent_children.erase(parent_children.begin() + i);
					break;
				}
			}
		}
		// remove from nodes
		nodes.erase(handle_to_delete);
	}

	std::string const beg_braket = "╭";
	std::string const end_braket = "╰";
	std::string const long_braket = "│││││││││││││││││││││││││││││││││││││││││││││";

	auto SceneTree::print2Console() noexcept -> void
	{
		printNode2Console(root, 0);
	}

	auto SceneTree::printNode2Console(SceneNodeHandle const& node, size_t bracket_num) noexcept -> void
	{
		std::string_view prefix{ long_braket.c_str(), bracket_num * 3};
		ECS::TagComponent& tag = nodes[node].entity.getComponent<ECS::TagComponent>();
		SE_CORE_INFO("{0}{1} {2} :: {3}", prefix, beg_braket, tag.Tag, nodes[node].uid);
		for (int i = 0; i < nodes[node].children.size(); i++)
		{
			printNode2Console(nodes[node].children[i], bracket_num + 1);
		}
		SE_CORE_INFO("{0}{1}", prefix, end_braket);
	}

	auto SceneTree::updateTransformForNode(SceneNodeHandle const& node, glm::mat4x4 const& precursor_transform) noexcept -> void
	{
		ECS::Entity entity = nodes[node].entity;
		auto& transform = entity.getComponent<GFX::Transform>();
		transform.propagateFromPrecursor(precursor_transform);

		for (int i = 0; i < nodes[node].children.size(); i++)
		{
			updateTransformForNode(nodes[node].children[i], transform.getAccumulativeTransform());
		}
	}

	auto SceneTree::updateTransforms() noexcept -> void
	{
		updateTransformForNode(root, glm::mat4x4(1.0f));
	}
}