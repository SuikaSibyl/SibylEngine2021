module;
#include <cstdint>
#include <filesystem>
export module GFX.Mesh;
import Core.Asset;
import Core.MemoryManager;
import Core.Buffer;
import Core.Cache;
import ECS.UID;
import RHI.ILogicalDevice;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;
import RHI.IDeviceGlobal;

namespace SIByL::GFX
{
	export struct Mesh :public Asset
	{
		Mesh(uint64_t const& uid, RHI::ILogicalDevice* logical_device);
		Mesh(Buffer* vb, Buffer* ib, RHI::ILogicalDevice* logical_device);

		MemScope<RHI::IVertexBuffer> vertexBuffer;
		MemScope<RHI::IIndexBuffer> indexBuffer;

	private:
		auto init(Buffer* vb, Buffer* ib, RHI::ILogicalDevice* logical_device) noexcept -> void;
	};

	struct MeshHeader
	{};

	Mesh::Mesh(uint64_t const& uid, RHI::ILogicalDevice* logical_device)
	{
		identifier = uid;
		MeshHeader header;
		Buffer vb, ib;
		Buffer* buffers[2] = { &vb,&ib };
		CacheBrain::instance()->loadCache(identifier, header, buffers);
		init(&vb, &ib, logical_device);
	}

	Mesh::Mesh(Buffer* vb, Buffer* ib, RHI::ILogicalDevice* logical_device)
	{
		init(vb, ib, logical_device);

		identifier = ECS::UniqueID::RequestUniqueID();
		Buffer* buffers[2] = { vb,ib };
		CacheBrain::instance()->saveCache(identifier, MeshHeader{}, buffers, 2, 0);
	}

	auto Mesh::init(Buffer* vb, Buffer* ib, RHI::ILogicalDevice* logical_device) noexcept -> void
	{
		auto* factory = RHI::DeviceToGlobal::getGlobal(logical_device)->getResourceFactory();
		vertexBuffer = factory->createVertexBuffer(vb);
		indexBuffer = factory->createIndexBuffer(ib, ib->getStride());
	}

}