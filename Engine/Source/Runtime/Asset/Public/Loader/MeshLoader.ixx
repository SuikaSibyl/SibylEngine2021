module;
#include <string>
#include <filesystem>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
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
		MeshLoader(Mesh& mesh) :mesh(mesh) {}
		MeshLoader(Mesh& mesh, RHI::IResourceFactory* factory, RuntimeAssetManager* manager)
			:DedicatedLoader(factory, manager), mesh(mesh) {}

		virtual auto loadFromFile(std::filesystem::path path) noexcept -> void override;
		virtual auto loadFromCache(uint64_t const& path) noexcept -> void override;
		virtual auto saveAsCache(uint64_t const& path) noexcept -> void override;

		Mesh& mesh;
		Buffer vb, ib;
	};

	auto MeshLoader::loadFromCache(uint64_t const& path) noexcept -> void
	{
		Buffer* buffers[2] = { &vb,&ib };
		CacheBrain::instance()->loadCache(path, mesh.desc, buffers);

		mesh.vertexBuffer = resourceFactory->createVertexBuffer(&vb);
		mesh.indexBuffer = resourceFactory->createIndexBuffer(&ib, ib.getStride());
	}

	auto MeshLoader::saveAsCache(uint64_t const& path) noexcept -> void
	{
		Buffer* buffers[2] = { &vb,&ib };
		CacheBrain::instance()->saveCache(path, mesh.desc, buffers, 2, 0);
	}

	export struct ExternalMeshSniffer
	{
		using Node = void*;
		auto loadFromFile(std::filesystem::path path) noexcept -> Node;
		auto interpretNode(Node node, uint32_t& mesh_num, uint32_t& children_num, std::string& name) noexcept -> void;
		auto fillVertexIndex(Node node, Buffer& vb, Buffer& ib) noexcept -> void;

		auto getNodeChildren(Node node, uint32_t index) noexcept -> Node;

		Assimp::Importer importer;
		aiScene const* scene = nullptr;
	};
}