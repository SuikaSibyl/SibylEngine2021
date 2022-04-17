module;
#include <cstdint>
#include <filesystem>
export module GFX.Mesh;
import Core.MemoryManager;
import Core.Buffer;
import Core.Cache;
import ECS.UID;
import RHI.ILogicalDevice;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;
import RHI.IDeviceGlobal;
import Asset.Asset;
import Asset.Mesh;
import Asset.AssetLayer;

namespace SIByL::GFX
{
	export struct Mesh
	{
		static auto query(Asset::GUID guid, Asset::AssetLayer* layer) noexcept -> Mesh;

		Asset::GUID guid = 0;
		Asset::MeshDesc meshDesc;
		RHI::IVertexBuffer* vertexBuffer;
		RHI::IIndexBuffer* indexBuffer;
	};

	auto Mesh::query(Asset::GUID guid, Asset::AssetLayer* layer) noexcept -> Mesh
	{
		Mesh mesh;
		Asset::Mesh* asset_mesh = layer->mesh(guid);
		mesh.guid = guid;
		mesh.meshDesc = asset_mesh->desc;
		mesh.vertexBuffer = asset_mesh->vertexBuffer.get();
		mesh.indexBuffer = asset_mesh->indexBuffer.get();
		return mesh;
	}
}