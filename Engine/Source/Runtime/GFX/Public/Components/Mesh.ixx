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

namespace SIByL::GFX
{
	export struct Mesh
	{
		Asset::GUID guid;
		RHI::IVertexBuffer* vertexBuffer;
		RHI::IIndexBuffer* indexBuffer;
	};
}