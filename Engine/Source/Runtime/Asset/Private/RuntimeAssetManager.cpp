module;
#include <map>
#include <string>
module Asset.RuntimeAssetManager;
import Asset.Asset;
import ECS.UID;

namespace SIByL::Asset
{
	// Manager Life Cycle

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

	auto RuntimeAssetManager::addNewAsset(ResourceItem const& item) noexcept -> GUID
	{
		GUID guid = ECS::UniqueID::RequestUniqueID();
		assetsMap.emplace(guid, item);
		return guid;
	}

	auto RuntimeAssetManager::serialize() noexcept -> void
	{

	}

	auto RuntimeAssetManager::deserialize() noexcept -> void
	{

	}
}