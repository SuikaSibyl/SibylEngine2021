module;

export module GFX.RDG.IndirectDrawBufferNode;
import Core.MemoryManager;
import RHI.IFactory;
import GFX.RDG.Common;
import GFX.RDG.StorageBufferNode;

namespace SIByL::GFX::RDG
{
	export struct IndirectDrawBufferNode :public StorageBufferNode
	{
	public:
		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override
		{
			size = sizeof(unsigned int) * 5;
			storageBuffer = factory->createIndirectDrawBuffer();
		}
	};
}