module;
#include <utility>
#include <string_view>
module Asset.AssetLayer;
import Core.Layer;
import Core.MemoryManager;
import RHI.IFactory;
import Asset.RuntimeAssetManager;
import Asset.Asset;
import Asset.DedicatedLoader;
import Asset.Mesh;
import Asset.MeshLoader;
import Asset.Texture;
import Asset.TextureLoader;
import Asset.RuntimeAssetLibrary;

namespace SIByL::Asset
{
	auto AssetLayer::mesh(GUID guid) noexcept -> Mesh*
	{
		Mesh* mesh = runtimeLibrary.meshLib.tryFind(guid);
		if (mesh == nullptr)
		{
			MemScope<Mesh> tmp_mesh = MemNew<Mesh>();
			MeshLoader loader(*(tmp_mesh.get()), resourceFactory, &runtimeManager);
			loader.fromGUID(guid);
			mesh= runtimeLibrary.meshLib.emplace(guid, std::move(tmp_mesh));
		}
		return mesh;
	}

	auto AssetLayer::texture(GUID guid) noexcept -> Texture*
	{
		Texture* texture = runtimeLibrary.textureLib.tryFind(guid);
		if (texture == nullptr)
		{
			MemScope<Texture> tmp_texture = MemNew<Texture>();
			TextureLoader loader(*(tmp_texture.get()), resourceFactory, &runtimeManager);
			loader.fromGUID(guid);
			texture = runtimeLibrary.textureLib.emplace(guid, std::move(tmp_texture));
		}
		return texture;
	}
}