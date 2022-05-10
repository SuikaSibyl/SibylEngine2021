module;
#include <cstdint>
#include <filesystem>
export module GFX.ParticleSystem;
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

namespace SIByL::GFX
{
	export struct ParticleSystem
	{
		bool showCluster = false;
		bool needRebuildPipeline = false;
	};
}