module;
#include <vector>
#include <unordered_map>
module GFX.RDG.ExternalAccess;
import RHI.IEnum;
import RHI.IMemoryBarrier;
import RHI.IMemoryBarrier;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;

namespace SIByL::GFX::RDG
{
	auto ExternalAccessPass::onCompile(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		barriers.clear();
		RenderGraph* rg = (RenderGraph*)graph;
		renderGraph = graph;

		for (auto iter : externalAccessMap)
		{
			auto handle = iter.second.resourceHandle;
			switch (rg->getResourceNode(handle)->type)
			{
			case NodeDetailedType::STORAGE_BUFFER:
			{
				rg->getResourceNode(handle)->getConsumeHistory().emplace_back
				(ConsumeHistory{ handle, ConsumeKind::BUFFER_READ_WRITE });
			}
			break;
			case NodeDetailedType::COLOR_TEXTURE:
			{
				TextureBufferNode* texture_node = rg->getTextureBufferNode(handle);
				texture_node->getConsumeHistory().emplace_back
				(ConsumeHistory{ this->handle, iter.second.consumeKind,
					texture_node->baseMipLevel,
					texture_node->levelCount,
					texture_node->baseArrayLayer,
					texture_node->layerCount });
			}
			break;
			default:
				break;
			}
		}
	}

	auto ExternalAccessPass::onCommandRecord(RHI::ICommandBuffer* commandbuffer, uint32_t flight) noexcept -> void
	{
		// push all barriers
		RenderGraph* render_graph = (RenderGraph*)renderGraph;
		for (auto barrier : barriers) commandbuffer->cmdPipelineBarrier(render_graph->barrierPool.getBarrier(barrier));
	}

	auto ExternalAccessPass::insertExternalAccessItem(ExternalAccessItem const& item) noexcept -> bool
	{
		if (externalAccessMap.find(item.resourceHandle) == externalAccessMap.end())
		{
			externalAccessMap.emplace(item.resourceHandle, item);
			return true;
		}
		else
			return false;
	}
}