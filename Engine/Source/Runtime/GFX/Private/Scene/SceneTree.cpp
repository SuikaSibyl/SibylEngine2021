module;
#include <vector>
#include <string>
#include <string_view>
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
		addNode("Scene", nullptr);
	}

	auto SceneTree::addNode(std::string const& name, SceneNode* parent) noexcept -> SceneNode*
	{
		ECS::Entity entity = context.createEntity(name);
		nodes.emplace_back(entity, parent);
		return &nodes.back();
	}

	auto SceneTree::moveNode(SceneNode* node, SceneNode* parent) noexcept -> void
	{
		// remove node from its previous parent
		SceneNode* prev_parent = node->parent;
		for (auto iter = prev_parent->children.begin(); iter != prev_parent->children.end();)
		{
			if ((*iter) == node)
			{
				iter = prev_parent->children.erase(iter);
				break;
			}
			else
				iter++;
		}
		// add node to its new parent
		parent->children.emplace_back(node);
		// change the node info
		node->parent = parent;
	}

	auto SceneTree::removeNode(SceneNode* node) noexcept -> void
	{
		if (node == &nodes[0]) return;
		// remove children
		context.destroyEntity(node->entity);
		for (int i = 0; i < node->children.size(); i++)
		{
			removeNode(node->children[i]);
		}
		// remove from nodes
		for (auto iter = nodes.begin(); iter != nodes.end();)
		{
			if (&(*iter) == node)
			{
				iter = nodes.erase(iter);
				break;
			}
			else
				iter++;
		}
	}

	std::string const beg_braket = "╭";
	std::string const end_braket = "╰";
	std::string const long_braket = "│││││││││││││││││││││││││││││││││││││││││││││";

	auto SceneTree::print2Console() noexcept -> void
	{
		printNode2Console(&nodes[0], 0);
	}

	auto SceneTree::printNode2Console(SceneNode* node, size_t bracket_num) noexcept -> void
	{
		std::string_view prefix{ long_braket.c_str(), bracket_num };
		ECS::TagComponent& tag = node->entity.GetComponent<ECS::TagComponent>();
		SE_CORE_INFO("{0}{1} {2}", prefix, beg_braket, tag.Tag);
		//for (int i = 0; i < node->children.size(); i++)
		//{
		//	printNode2Console(node->children[i], bracket_num + 1);
		//}
		SE_CORE_INFO("{0}{1}", prefix, end_braket);
	}

}