module;
#include <utility>
#include <vector>
#include <filesystem>
#include <string>
#include <unordered_map>
#include "entt/entt.hpp"
#include "yaml-cpp/yaml.h"
#include "yaml-cpp/node/node.h"
module GFX.Scene;
import Core.File;
import Core.Log;
import ECS.Entity;
import ECS.TagComponent;
import GFX.SceneTree;
import GFX.Mesh;
import RHI.ILogicalDevice;

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

		if (entity.hasComponent<GFX::Mesh>())
		{
			out << YAML::Key << "MeshComponent";
			auto& mesh = entity.getComponent<GFX::Mesh>();
			out << YAML::Value << YAML::BeginMap;
			out << YAML::Key << "uid" << YAML::Value << mesh.getIdentifier();
			out << YAML::Key << "path" << YAML::Value << mesh.getAttachedPath().string();
			out << YAML::EndMap;
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
		if (node->children.size() > 0)
		{
			out << YAML::Key << "children" << YAML::Value << YAML::BeginSeq;
			for (int i = 0; i < node->children.size(); i++)
				out << node->children[i];
			out << YAML::EndSeq;
		}
		// components
		out << YAML::Key << "components" << YAML::Value;
		serializeEntity(out, node->entity);
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

		out << YAML::Key << "SceneEnd" << YAML::Value << "TRUE";
		out << YAML::EndMap;
		AssetLoader* scene_loader = getSceneLoader();
		Buffer scene_proxy((void*)out.c_str(), out.size(), 1);
		scene_loader->syncWriteAll(path, scene_proxy);
	}

	auto deserializeEntity(YAML::NodeAoS& components, ECS::Entity& entity, RHI::ILogicalDevice* device) noexcept -> void
	{
		// Mesh Component
		// -----------------------------------------------
		auto meshComponent = components["MeshComponent"];
		if (meshComponent)
		{
			std::string path = meshComponent["path"].as<std::string>();
			uint64_t uid = meshComponent["uid"].as<uint64_t>();
			if ((uid != 0) && (path.length() == 0))
			{
				auto& mc = entity.addComponent<GFX::Mesh>(uid, device);
			}
			else
			{
				SE_CORE_ERROR("GFX :: Scene Deserialize :: mesh component with path not supported yet (TODO)");
			}
		}
	}

	auto Scene::deserialize(std::filesystem::path path, RHI::ILogicalDevice* device) noexcept -> void
	{
		AssetLoader* scene_loader = getSceneLoader();
		Buffer scene_buffer;
		scene_loader->syncReadAll(path, scene_buffer);
		YAML::NodeAoS data = YAML::Load(scene_buffer.getData());

		// check scene name
		if (!data["SceneName"])
		{
			SE_CORE_ERROR("GFX :: Scene Name not found when deserializing {0}", path.string());
		} 
		if (!data["SceneNodes"])
		{
			SE_CORE_ERROR("GFX :: Scene Nodes not found when deserializing {0}", path.string());
		}
		
		auto scene_nodes = data["SceneNodes"];
		for (auto node : scene_nodes)
		{
			uint64_t uid = node["uid"].as<uint64_t>();
			uint64_t parent = node["parent"].as<uint64_t>();
			auto components = node["components"];
			auto tagComponent = components["TagComponent"].as<std::string>();
			auto children = node["children"];
			std::vector<uint64_t> children_uids(children.size());
			uint32_t idx = 0;
			if (children)
				for (auto child : children)
					children_uids[idx++] = child.as<uint64_t>();
			tree.addNode(tagComponent, uid, parent, std::move(children_uids));

			deserializeEntity(components, tree.nodes[uid].entity, device);

			if (parent == 0)
				tree.appointRoot(uid);
		}
	}
}