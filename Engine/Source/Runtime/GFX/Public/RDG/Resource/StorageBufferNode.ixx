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
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override
		{
			if (!(attributes & (uint32_t)NodeAttrbutesFlagBits::PLACEHOLDER))
			{
				storageBuffer = factory->createStorageBuffer(size);
			}
		}

		virtual auto getStorageBuffer() noexcept -> RHI::IStorageBuffer*
		{
			if (!(attributes & (uint32_t)NodeAttrbutesFlagBits::PLACEHOLDER))
			{
				return storageBuffer.get();
			}
			return externalStorageBuffer;
		}

		size_t size;
		RHI::IStorageBuffer* externalStorageBuffer = nullptr;
		MemScope<RHI::IStorageBuffer> storageBuffer = nullptr;
	};
}