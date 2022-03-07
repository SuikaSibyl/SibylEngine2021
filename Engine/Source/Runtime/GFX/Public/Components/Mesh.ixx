export module GFX.Mesh;
import Core.MemoryManager;
import Core.Buffer;
import RHI.ILogicalDevice;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;
import RHI.IDeviceGlobal;

namespace SIByL::GFX
{
	export struct Mesh
	{
		Mesh(Buffer* vb, Buffer* ib, RHI::ILogicalDevice* logical_device);

		MemScope<RHI::IVertexBuffer> vertexBuffer;
		MemScope<RHI::IIndexBuffer> indexBuffer;
	};

	Mesh::Mesh(Buffer* vb, Buffer* ib, RHI::ILogicalDevice* logical_device)
	{
		auto* factory = RHI::DeviceToGlobal::getGlobal(logical_device)->getResourceFactory();
		vertexBuffer = factory->createVertexBuffer(vb);
		indexBuffer = factory->createIndexBuffer(ib, ib->getStride());
	}

}