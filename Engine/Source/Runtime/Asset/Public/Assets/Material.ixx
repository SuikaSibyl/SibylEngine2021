module;
#include <cstdint>
#include <vector>
export module Asset.Material;
import Asset.Asset;
import Core.Buffer;
import Core.MemoryManager;
import RHI.IVertexBuffer;
import RHI.IIndexBuffer;

namespace SIByL::Asset
{
	export struct Material :public Asset
	{
		std::vector<GUID> sampled_textures;
	};
}