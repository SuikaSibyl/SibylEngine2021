module;
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <algorithm>
#include "entt/entt.hpp"
module GFX.SceneTree;
import Core.Log;
import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;

namespace SIByL::GFX
{
	SceneTree::SceneTree()
	{
		root = addNode("Scene", 0);
	}

	auto SceneTree::addNode(std::string const& name, SceneNodeHandle const& parent) noexcept -> SceneNodeHandle
	{
		ECS::Entity entity = context.createEntity(name);
		uint64_t uid = ECS::UniqueID::RequestUniqueID();
		nodes[uid] = SceneNode{ entity, uid, parent, {} };

		// add children to parent
		if (parent != 0) // if not root
			nodes[parent].children.emplace_back(uid);

		return uid;
	}

	auto SceneTree::moveNode(uint64_t const& handle, uint64_t const& parent) noexcept -> void
	{
		// remove node from its previous parent
		// remove from parent
		auto& parent_children = nodes[nodes[handle].parent].children;
		for (int i = 0; i < parent_children.size(); i++)
		{
			if (parent_children[i] = handle)
			{
				parent_children.erase(parent_children.begin() + i);
				break;
			}
		}
		// add node to its new parent
		nodes[parent].children.emplace_back(handle);
		// change the node info
		nodes[handle].parent = parent;
	}

	auto SceneTree::removeNode(uint64_t const& handle) noexcept -> void
	{
		if (handle == root) return;
		// remove children
		context.destroyEntity(nodes[handle].entity);
		for (int i = 0; i < nodes[handle].children.size(); i++)
		{
			removeNode(nodes[handle].children[i]);
		}
		// remove from parent
		auto& parent_children = nodes[nodes[handle].parent].children;
		for (int i = 0; i < parent_children.size(); i++)
		{
			if (parent_children[i] = handle)
			{
				parent_children.erase(parent_children.begin() + i);
				break;
			}
		}
		// remove from nodes
		auto iter = nodes.find(handle);
		if (iter != nodes.end()) iter = nodes.erase(iter);
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
}