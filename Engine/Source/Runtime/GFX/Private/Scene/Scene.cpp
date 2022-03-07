module;
#include <vector>
#include <filesystem>
#include <string>
#include <unordered_map>
#include "entt/entt.hpp"
#include "yaml-cpp/yaml.h"
#include "yaml-cpp/node/node.h"
module GFX.Scene;
import Core.File;
import ECS.Entity;
import ECS.TagComponent;
import GFX.SceneTree;

namespace SIByL::GFX
{
	bool sceneLoaderInited = false;
	auto getSceneLoader() noexcept -> AssetLoader*
	{
		static AssetLoader sceneLoader;
		if (sceneLoaderInited == false)
		{
			sceneLoader.addSearchPath("./assets/");
			sceneLoaderInited = true;
		}
		return &sceneLoader;
	}

	auto serializeEntity(YAML::Emitter& out, ECS::Entity& entity) noexcept -> void
	{
		out << YAML::BeginMap;

		if (entity.hasComponent<ECS::TagComponent>())
		{
			out << YAML::Key << "TagComponent";
			auto& tag = entity.getComponent<ECS::TagComponent>().Tag;
			out << YAML::Value << tag;
		}

		out << YAML::EndMap;
	}

	auto serializeSceneNode(YAML::Emitter& out, SceneNode* node) noexcept -> void
	{
		out << YAML::BeginMap;
		// uid
		out << YAML::Key << "uid" << YAML::Value << node->uid;
		// parent
		out << YAML::Key << "parent" << YAML::Value << node->parent;
		// children
		out << YAML::Key << "children" << YAML::Value << YAML::BeginSeq;
		for (int i = 0; i < node->children.size(); i++)
			out << node->children[i];
		out << YAML::EndSeq;
		// components
		out << YAML::Key << "components" << YAML::Value << YAML::BeginSeq;
		//serializeEntity(out, node->entity);
		out << YAML::EndSeq;
		// end
		out << YAML::EndMap;
	}

	auto Scene::serialize(std::filesystem::path path) noexcept -> void
	{
		YAML::Emitter out;
		out << YAML::BeginMap;
		out << YAML::Key << "SceneName" << YAML::Value << tree.getNodeEntity(tree.root).getComponent<ECS::TagComponent>().Tag;

		out << YAML::Key << "SceneNodes" << YAML::Value << YAML::BeginSeq;
		for (auto iter = tree.nodes.begin(); iter != tree.nodes.end(); iter++)
			serializeSceneNode(out, &(iter->second));

		out << YAML::EndSeq;

		out << YAML::EndMap;
		AssetLoader* scene_loader = getSceneLoader();
		Buffer scene_proxy((void*)out.c_str(), out.size(), 1);
		scene_loader->syncWriteAll(path, scene_proxy);
	}

}