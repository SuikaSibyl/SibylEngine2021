module;
#include <cstdint>
#include <utility>
module GFX.RDG.StorageBufferNode;
import Core.Log;
import Core.MemoryManager;
import Core.BitFlag;
import RHI.IFactory;
import RHI.IStorageBuffer;
import GFX.RDG.Common;
import GFX.RDG.RenderGraph;
import RHI.IBarrier;
import RHI.IMemoryBarrier;

namespace SIByL::GFX::RDG
{
	auto StorageBufferNode::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;

		if (!(attributes & (uint32_t)NodeAttrbutesFlagBits::PLACEHOLDER))
		{
			storageBuffer = factory->createStorageBuffer(size);
		}

		if (consumeHistory.size() > 1)
		{
			unsigned int history_size = consumeHistory.size();
			unsigned int i_minus = history_size - 1;

			for (int i = 0; i < history_size; i++)
			{
				int left = i_minus;
				int right = i;
				// - A - BEGIN - B -
				//   |     | 
				// we create a A-B barrier in BEGIN
				if (consumeHistory[right].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN)
				{
					right = right + 1;
				}
				// - BEGIN - A - B - C - END -
				//     |     |
				// we create a C-A barrier in A
				if (consumeHistory[left].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN)
				{
					while (consumeHistory[left + 1].kind != ConsumeKind::MULTI_DISPATCH_SCOPE_END)
					{
						left++;
					}
				}
				// - A - BEGIN - B - C - END - D
				//                        |    |
				// we create a C-D barrier in D
				if (consumeHistory[left].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_END)
				{
					left--;
				}
				// - A - BEGIN - B - C - END - D
				//                   |    |   
				// we create a A-D barrier in End ( used only when 0 dispatch happens)
				if (consumeHistory[right].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_END)
				{
					while (consumeHistory[left + 1].kind != ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN)
					{
						left--;
					}
					right = right + 1;
				}

				if (left < 0) left += consumeHistory.size();
				if (right >= consumeHistory.size()) right -= consumeHistory.size();

				while (consumeHistory[left].kind >= ConsumeKind::SCOPE) left--;
				while (consumeHistory[right].kind >= ConsumeKind::SCOPE) right++;

				RHI::PipelineStageFlags srcStageMask = 0;
				RHI::PipelineStageFlags dstStageMask = 0;
				RHI::AccessFlags srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
				RHI::AccessFlags dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;

				if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::COMPUTE_PASS)
					srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
				else if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::RASTER_PASS)
					srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT
								 | (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;
								 //| (uint32_t)RHI::PipelineStageFlagBits::MESH_SHADER_BIT_NV;
				else SE_CORE_ERROR("RDG :: STORAGE BUFFER unknown access switch!");

				if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::COMPUTE_PASS)
					dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
				else if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::RASTER_PASS)
					dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT
					| (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;
				//| (uint32_t)RHI::PipelineStageFlagBits::MESH_SHADER_BIT_NV;
				else SE_CORE_ERROR("RDG :: STORAGE BUFFER unknown access switch!");

				MemScope<RHI::IBufferMemoryBarrier> buffer_memory_barrier = factory->createBufferMemoryBarrier({
					getStorageBuffer()->getIBuffer(),
					srcAccessFlags, //AccessFlags srcAccessMask;
					dstAccessFlags, //AccessFlags dstAccessMask;
					});

				MemScope<RHI::IBarrier> barrier = factory->createBarrier({
					srcStageMask,//srcStageMask
					dstStageMask,//dstStageMask
					0,
					{},
					{buffer_memory_barrier.get()},
					{}
					});

				BarrierHandle barrier_handle = rg->barrierPool.registBarrier(std::move(barrier));
				rg->getPassNode(consumeHistory[i].pass)->barriers.emplace_back(barrier_handle);
				i_minus = i;
			}
		}
	}

	auto StorageBufferNode::getStorageBuffer() noexcept -> RHI::IStorageBuffer*
	{
		if (!(attributes & (uint32_t)NodeAttrbutesFlagBits::PLACEHOLDER))
		{
			return storageBuffer.get();
		}
		return externalStorageBuffer;
	}
}