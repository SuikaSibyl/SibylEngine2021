module;
#include <vector>
export module GFX.RDG.UniformBufferNode;
import Core.MemoryManager;
import RHI.IFactory;
import RHI.IUniformBuffer;
import GFX.RDG.Common;
import GFX.RDG.ResourceNode;

namespace SIByL::GFX::RDG
{
	export struct UniformBufferNode :public ResourceNode
	{
	public:
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override
		{
			uniformBuffer = factory->createUniformBuffer(size);
		}

		size_t size;
		MemScope<RHI::IUniformBuffer> uniformBuffer = nullptr;
	};
}