module;
#include <map>
#include <string>
export module Asset.RuntimeAssetManager;
import Asset.Asset;

namespace SIByL::Asset
{
	export struct ResourceItem
	{
		std::string path;
	};

	export class RuntimeAssetManager
	{
	public:
		// Manager Life Cycle
		auto initialize() noexcept -> void;
		auto flush() noexcept -> void;
		auto exit() noexcept -> void;

		// Manager Functions
		auto addNewAsset(ResourceItem const& item) noexcept -> GUID;

		auto getAssetMap() const noexcept -> std::map<GUID, ResourceItem> const& { return assetsMap; }

	private:
		auto serialize() noexcept -> void;
		auto deserialize() noexcept -> void;
		std::map<GUID, ResourceItem> assetsMap;
	};
}