module;
#include <string>
#include <filesystem>
export module Asset.MeshLoader;
import Core.Buffer;
import Core.Cache;
import Core.MemoryManager;
import Asset.Asset;
import Asset.DedicatedLoader;
import Asset.Mesh;
import RHI.IFactory;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;

namespace SIByL::Asset
{
	export struct MeshLoader :public DedicatedLoader
	{
		MeshLoader(Mesh& mesh, RHI::IResourceFactory* factory, RuntimeAssetManager* manager)
			:DedicatedLoader(factory, manager), mesh(mesh) {}

		virtual auto loadFromFile(std::filesystem::path path) noexcept -> void override;
		virtual auto loadFromCache(uint64_t const& path) noexcept -> void override;
		virtual auto saveAsCache(uint64_t const& path) noexcept -> void override;

		Mesh& mesh;
		Buffer vb, ib;
	};

	struct MeshHeader
	{};

	auto MeshLoader::loadFromCache(uint64_t const& path) noexcept -> void
	{
		MeshHeader header;
		Buffer* buffers[2] = { &vb,&ib };
		CacheBrain::instance()->loadCache(path, header, buffers);

		mesh.vertexBuffer = resourceFactory->createVertexBuffer(&vb);
		mesh.indexBuffer = resourceFactory->createIndexBuffer(&ib, ib.getStride());
	}

	auto MeshLoader::saveAsCache(uint64_t const& path) noexcept -> void
	{
		Buffer* buffers[2] = { &vb,&ib };
		CacheBrain::instance()->saveCache(path, MeshHeader{}, buffers, 2, 0);
	}
}