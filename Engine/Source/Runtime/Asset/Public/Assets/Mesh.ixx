module;
#include <cstdint>
export module Asset.Mesh;
import Asset.Asset;
import Core.Buffer;
import Core.MemoryManager;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;

namespace SIByL::Asset
{
	export enum struct VertexInfoBits :uint32_t
	{
		POSITION = 1 << 0,
		COLOR = 1 << 1,
		UV = 1 << 2,
		NORMAL = 1 << 3,
		TANGENT = 1 << 4,
	};

	export using VertexInfoFlags = uint32_t;

	export struct MeshDesc
	{
		VertexInfoFlags vertexInfo;
		uint32_t subMeshID = 0;
	};

	export struct Mesh :public Asset
	{
		MeshDesc desc;
		MemScope<RHI::IVertexBuffer> vertexBuffer;
		MemScope<RHI::IIndexBuffer> indexBuffer;
	};
}