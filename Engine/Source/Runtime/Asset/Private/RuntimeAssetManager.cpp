module;
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include "yaml-cpp/yaml.h"
#include "yaml-cpp/node/node.h"
module Asset.RuntimeAssetManager;
import Asset.Asset;
import Core.File;
import Core.Log;
import ECS.UID;

namespace SIByL::Asset
{
	// Manager Life Cycle
	RuntimeAssetManager::RuntimeAssetManager()
		:assetLoader({ "./assets/" })
	{}
	
	auto RuntimeAssetManager::initialize() noexcept -> void
	{
		deserialize();
	}

	auto RuntimeAssetManager::flush() noexcept -> void
	{
		serialize();
	}

	auto RuntimeAssetManager::exit() noexcept -> void
	{
		serialize();
	}

	// Manager Functions
	auto RuntimeAssetManager::addNewAssetWithGUID(ResourceItem const& item, GUID const& guid) noexcept -> void
	{
		assetsMap.emplace(guid, item);
	}

	auto RuntimeAssetManager::addNewAsset(ResourceItem const& item) noexcept -> GUID
	{
		GUID guid = ECS::UniqueID::RequestUniqueID();
		assetsMap.emplace(guid, item);
		if (item.path != std::filesystem::path{})
		{
			inverseMap.emplace(item.path.c_str(), guid);
		}
		return guid;
	}

	auto RuntimeAssetManager::findAsset(GUID const& guid) noexcept -> ResourceItem*
	{
		auto iter = assetsMap.find(guid);
		if (iter == assetsMap.end()) return nullptr;
		return &(iter->second);
	}

	auto RuntimeAssetManager::serialize() noexcept -> void
	{
		{
			YAML::Emitter out;
			out << YAML::BeginMap;
			out << YAML::Key << "AssetMap" << YAML::Value << YAML::BeginSeq;
			for (auto iter : assetsMap)
			{
				out << YAML::BeginMap;
				out << YAML::Key << "GUID" << YAML::Value << iter.first;
				out << YAML::Key << "path" << YAML::Value << iter.second.path;
				out << YAML::Key << "kind" << YAML::Value << (uint64_t)iter.second.kind;
				out << YAML::Key << "cacheST" << YAML::Value << iter.second.cachedTime;
				out << YAML::Key << "cache" << YAML::Value << iter.second.cacheID;
				out << YAML::EndMap;
			}
			out << YAML::EndSeq;
			out << YAML::Key << "End" << YAML::Value << true;
			out << YAML::EndMap;
			// Output
			Buffer scene_proxy((void*)out.c_str(), out.size(), 1);
			assetLoader.syncWriteAll(".adb", scene_proxy);
		}

		{
			YAML::Emitter out;
			out << YAML::BeginMap;
			out << YAML::Key << "InverseMap" << YAML::Value << YAML::BeginSeq;
			for (auto iter : inverseMap)
			{
				out << YAML::BeginMap;
				out << YAML::Key << "path" << YAML::Value << iter.first;
				out << YAML::Key << "GUID" << YAML::Value << iter.second;
				out << YAML::EndMap;
			}
			out << YAML::EndSeq;
			out << YAML::Key << "End" << YAML::Value << true;
			out << YAML::EndMap;
			// Output
			Buffer scene_proxy((void*)out.c_str(), out.size(), 1);
			assetLoader.syncWriteAll(".iadb", scene_proxy);
		}
	}

	auto RuntimeAssetManager::deserialize() noexcept -> void
	{
		{
			Buffer asset_db_buffer;
			assetLoader.syncReadAll(".adb", asset_db_buffer);
			YAML::NodeAoS data = YAML::Load(asset_db_buffer.getData());

			// check scene name
			if (!data["AssetMap"])
			{
				SE_CORE_ERROR("Asset :: Asset Database Lost");
			}
			if (!data["End"])
			{
				SE_CORE_ERROR("Asset :: Asset Database not End Normally");
			}

			auto assets_nodes = data["AssetMap"];
			for (auto node : assets_nodes)
			{
				ResourceItem item;
				GUID guid = node["GUID"].as<uint64_t>();
				item.kind = (ResourceKind)node["kind"].as<uint64_t>();
				item.cachedTime = node["cacheST"].as<uint64_t>();
				item.cacheID = node["cache"].as<uint64_t>();
				item.path = node["path"].as<std::string>();
				assetsMap.emplace(guid, item);
			}
		}
		{
			Buffer asset_idb_buffer;
			assetLoader.syncReadAll(".iadb", asset_idb_buffer);
			YAML::NodeAoS data = YAML::Load(asset_idb_buffer.getData());

			// check scene name
			if (!data["InverseMap"])
			{
				SE_CORE_ERROR("Asset :: Inverse Database Lost");
			}
			if (!data["End"])
			{
				SE_CORE_ERROR("Asset :: Inverse Database not End Normally");
			}

			auto assets_nodes = data["InverseMap"];
			for (auto node : assets_nodes)
			{
				GUID guid = node["GUID"].as<uint64_t>();
				std::string path = node["path"].as<std::string>();
				inverseMap.emplace(path, guid);
			}
		}
	}
}