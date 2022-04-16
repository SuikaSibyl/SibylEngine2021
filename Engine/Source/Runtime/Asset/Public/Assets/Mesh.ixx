export module Asset.Mesh;
import Asset.Asset;
import Core.Buffer;
import Core.MemoryManager;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;

namespace SIByL::Asset
{
	export struct Mesh :public Asset
	{
		MemScope<RHI::IVertexBuffer> vertexBuffer;
		MemScope<RHI::IIndexBuffer> indexBuffer;
	};
}