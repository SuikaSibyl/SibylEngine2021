module;

export module GFX.RDG.StorageBufferNode;
import Core.MemoryManager;
import RHI.IFactory;
import RHI.IStorageBuffer;
import GFX.RDG.Common;
import GFX.RDG.ResourceNode;

namespace SIByL::GFX::RDG
{
	export struct StorageBufferNode :public ResourceNode
	{
	public:
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override
		{
			storageBuffer = factory->createStorageBuffer(size);
		}

		size_t size;
		MemScope<RHI::IStorageBuffer> storageBuffer = nullptr;
	};
}