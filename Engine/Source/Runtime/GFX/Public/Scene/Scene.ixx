module;
#include <filesystem>
export module GFX.Scene;
import GFX.SceneTree;
import Core.File;
import Asset.AssetLayer;

namespace SIByL::GFX
{
	export class Scene
	{
	public:
		auto serialize(std::filesystem::path path) noexcept -> void;
		auto deserialize(std::filesystem::path path, Asset::AssetLayer* asset_layer) noexcept -> void;

		SceneTree tree;
	};
}