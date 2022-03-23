module;
#include <cstdint>
export module GFX.RDG.StorageBufferNode;
import Core.MemoryManager;
import RHI.IFactory;
import RHI.IStorageBuffer;
import GFX.RDG.Common;

namespace SIByL::GFX::RDG
{
	export struct StorageBufferNode :public ResourceNode
	{
	public:
		StorageBufferNode() { type = NodeDetailedType::STORAGE_BUFFER; }

		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto getStorageBuffer() noexcept -> RHI::IStorageBuffer*;

		size_t size;
		RHI::IStorageBuffer* externalStorageBuffer = nullptr;
		MemScope<RHI::IStorageBuffer> storageBuffer = nullptr;
	};
}