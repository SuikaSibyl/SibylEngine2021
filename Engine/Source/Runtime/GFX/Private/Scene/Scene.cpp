module;
#include <utility>
#include <vector>
#include <filesystem>
#include <string>
#include <unordered_map>
#include "entt/entt.hpp"
#include "glm/glm.hpp"
#include "yaml-cpp/yaml.h"
#include "yaml-cpp/node/node.h"
module GFX.Scene;
import Core.File;
import Core.Log;
import ECS.Entity;
import ECS.TagComponent;
import GFX.SceneTree;
import GFX.Mesh;
import GFX.Camera;
import GFX.Transform;
import GFX.Serializer;
import GFX.Renderer;
import RHI.ILogicalDevice;
import Asset.AssetLayer;
import Asset.Mesh;

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

		if (entity.hasComponent<GFX::Transform>())
		{
			out << YAML::Key << "Transform";
			Transform& tansform = entity.getComponent<GFX::Transform>();
			out << YAML::Value << YAML::BeginMap;
			glm::vec3 translation = tansform.getTranslation();
			out << YAML::Key << "translation" << YAML::Value << translation;
			out << YAML::Key << "eulerAngles" << YAML::Value << tansform.getEulerAngles();
			out << YAML::Key << "scale" << YAML::Value << tansform.getScale();
			out << YAML::EndMap;
		}

		if (entity.hasComponent<GFX::Camera>())
		{
			out << YAML::Key << "Camera";
			auto& camera = entity.getComponent<GFX::Camera>();
			out << YAML::Value << YAML::BeginMap;
			out << YAML::Key << "fov" << YAML::Value << camera.getFovy();
			out << YAML::Key << "aspect" << YAML::Value << camera.getAspect();
			out << YAML::Key << "near" << YAML::Value << camera.getNear();
			out << YAML::Key << "far" << YAML::Value << camera.getFar();
			out << YAML::EndMap;
		}

		if (entity.hasComponent<GFX::Mesh>())
		{
			out << YAML::Key << "MeshComponent";
			auto& mesh = entity.getComponent<GFX::Mesh>();
			out << YAML::Value << YAML::BeginMap;
			out << YAML::Key << "guid" << YAML::Value << mesh.guid;
			out << YAML::Key << "vert" << YAML::Value << mesh.meshDesc.vertexInfo;
			out << YAML::EndMap;
		}

		if (entity.hasComponent<GFX::Renderer>())
		{
			out << YAML::Key << "Renderer";
			auto& renderer = entity.getComponent<GFX::Renderer>();
			out << YAML::Value << YAML::BeginMap;
			// subRenderer
			out << YAML::Key << "SubRenderer" << YAML::Value << YAML::BeginSeq;
			for (int i = 0; i < renderer.subRenderers.size(); i++)
			{
				auto& subRenderer = renderer.subRenderers[i];
				out << YAML::BeginMap;
				out << YAML::Key << "PassHandle" << YAML::Value << subRenderer.pass;
				out << YAML::EndMap;
			}
			out << YAML::EndSeq;

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

	auto deserializeEntity(YAML::NodeAoS& components, ECS::Entity& entity, Asset::AssetLayer* asset_layer) noexcept -> void
	{
		// Transform Component
		// -----------------------------------------------
		auto transform = components["Transform"];
		if (transform)
		{
			glm::vec3 translation = transform["translation"].as<glm::vec3>();
			glm::vec3 eulerAngles = transform["eulerAngles"].as<glm::vec3>();
			glm::vec3 scale = transform["scale"].as<glm::vec3>();
			auto& tc = entity.getComponent<GFX::Transform>();
			tc.setTranslation(translation);
			tc.setEulerAngles(eulerAngles);
			tc.setScale(scale);
		}

		// Camera Component
		// -----------------------------------------------
		auto camera = components["Camera"];
		if (camera)
		{
			float fov = camera["fov"].as<float>();
			float aspect = camera["aspect"].as<float>();
			float near = camera["near"].as<float>();
			float far = camera["far"].as<float>();

			auto& cc = entity.addComponent<GFX::Camera>();
			cc.setFovy(fov);
			cc.setAspect(aspect);
			cc.setNear(near);
			cc.setFar(far);
		}

		// Mesh Component
		// -----------------------------------------------
		auto meshComponent = components["MeshComponent"];
		if (meshComponent)
		{
			uint64_t guid = meshComponent["guid"].as<uint64_t>();
			uint32_t desc = meshComponent["vert"].as<uint32_t>();
			auto& mc = entity.addComponent<GFX::Mesh>();
			mc.guid = guid;
			mc.meshDesc = { (Asset::VertexInfoFlags)desc };
			Asset::Mesh* asset_mesh = asset_layer->mesh(guid);
			mc.vertexBuffer = asset_mesh->vertexBuffer.get();
			mc.indexBuffer = asset_mesh->indexBuffer.get();
			if (mc.meshDesc.vertexInfo != asset_mesh->desc.vertexInfo) SE_CORE_ERROR("GFX :: Scene Deserialize :: Mesh Component Deserialize failed :: Scene Vertex Desc != Cache Vertex Desc");
		}

		auto rendererComponent = components["Renderer"];
		if (rendererComponent)
		{
			auto subRenderers = rendererComponent["SubRenderer"];
			auto& rc = entity.addComponent<GFX::Renderer>();
			if (subRenderers)
				for (auto sub : subRenderers)
				{
					uint64_t pass_handle = sub["PassHandle"].as<uint64_t>();
					rc.subRenderers.emplace_back(pass_handle);
				}
		}
	}

	auto Scene::deserialize(std::filesystem::path path, Asset::AssetLayer* asset_layer) noexcept -> void
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

			deserializeEntity(components, tree.nodes[uid].entity, asset_layer);

			if (parent == 0)
				tree.appointRoot(uid);
		}
	}
}