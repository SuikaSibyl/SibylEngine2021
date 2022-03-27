module;
#include <cstdint>
#include <vector>
module GFX.RDG.ColorBufferNode;
import GFX.RDG.Common;
import Core.Log;
import Core.BitFlag;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.IFactory;
import RHI.IBarrier;
import RHI.IMemoryBarrier;
import GFX.RDG.RenderGraph;
import GFX.RDG.TextureBufferNode;

namespace SIByL::GFX::RDG
{
	ColorBufferNode::ColorBufferNode(RHI::ResourceFormat _format, float const& rel_width, float const& rel_height)
	{
		relWidth = rel_width;
		relHeight = rel_height;
		format = _format;
		type = NodeDetailedType::COLOR_TEXTURE;
	}
	
	auto ColorBufferNode::devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		// Create Actual Texture Resource
		if (!hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
		{
			RenderGraph* render_graph = (RenderGraph*)graph;
			texture.scope = factory->createTexture(
				{
				RHI::ResourceType::Texture2D, //ResourceType type;
				format, //ResourceFormat format;
				RHI::ImageTiling::OPTIMAL, //ImageTiling tiling;
				(uint32_t)RHI::ImageUsageFlagBits::SAMPLED_BIT | (uint32_t)RHI::ImageUsageFlagBits::TRANSFER_SRC_BIT | usages, //ImageUsageFlags usages;
				RHI::BufferShareMode::EXCLUSIVE, //BufferShareMode shareMode;
				RHI::SampleCount::COUNT_1_BIT, //SampleCount sampleCount;
				RHI::ImageLayout::UNDEFINED, //ImageLayout layout;
				(uint32_t)(render_graph->getDatumWidth() * relWidth), //uint32_t width;
				(uint32_t)(render_graph->getDatumHeight() * relHeight) //uint32_t height;
				});
			textureView.scope = factory->createTextureView(getTexture(), usages);
		}
	}

	auto ColorBufferNode::rereference(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		devirtualize(graph, factory);
	}

	auto ColorBufferNode::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* rg = (RenderGraph*)graph;

		if (consumeHistory.size() > 1)
		{
			if (consumeHistory[0].kind >= ConsumeKind::SCOPE && consumeHistory.size() == 3) return;

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
				RHI::ImageLayout oldLayout = RHI::ImageLayout::UNDEFINED;
				RHI::ImageLayout newLayout = RHI::ImageLayout::UNDEFINED;
				RHI::AccessFlags srcAccessFlags = 0;
				RHI::AccessFlags dstAccessFlags = 0;

				// STORE_READ_WRITE -> RENDER_TARGET
				if (consumeHistory[left].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE && consumeHistory[right].kind == ConsumeKind::RENDER_TARGET)
				{
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::COMPUTE_PASS)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> RENDER_TARGET, STORAGE_READ_WRITE is not consumed by a compute pass!");

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::RASTER_PASS)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> RENDER_TARGET, RENDER_TARGET is not consumed by a raster pass!");

					oldLayout = RHI::ImageLayout::GENERAL;
					newLayout = RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT;
				}
				// RENDER_TARGET -> STORE_READ_WRITE
				else if (consumeHistory[left].kind == ConsumeKind::RENDER_TARGET && consumeHistory[right].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE)
				{
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::RASTER_PASS)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT;
					else SE_CORE_ERROR("RDG :: RENDER_TARGET -> STORAGE_READ_WRITE, RENDER_TARGET is not consumed by a raster pass!");

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::COMPUTE_PASS)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: RENDER_TARGET -> STORAGE_READ_WRITE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					oldLayout = RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
					newLayout = RHI::ImageLayout::GENERAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
				}
				// STORE_READ_WRITE -> STORE_READ_WRITE
				else if (consumeHistory[left].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE && consumeHistory[right].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE)
				{
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::COMPUTE_PASS)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> STORAGE_READ_WRITE, RENDER_TARGET is not consumed by a compute pass!");

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::COMPUTE_PASS)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> STORAGE_READ_WRITE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					oldLayout = RHI::ImageLayout::GENERAL;
					newLayout = RHI::ImageLayout::GENERAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
				}
				// STORE_READ_WRITE -> IMAGE_SAMPLE
				else if (consumeHistory[left].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE && consumeHistory[right].kind == ConsumeKind::IMAGE_SAMPLE)
				{
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::COMPUTE_PASS)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> IMAGE_SAMPLE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::COMPUTE_PASS)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::RASTER_PASS)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT 
									 | (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT
									 | (uint32_t)RHI::PipelineStageFlagBits::MESH_SHADER_BIT_NV;
					oldLayout = RHI::ImageLayout::GENERAL;
					newLayout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
				}
				// IMAGE_SAMPLE -> STORE_READ_WRITE
				else if (consumeHistory[left].kind == ConsumeKind::IMAGE_SAMPLE && consumeHistory[right].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE)
				{
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::COMPUTE_PASS)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::RASTER_PASS)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT
									 | (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT
									 | (uint32_t)RHI::PipelineStageFlagBits::MESH_SHADER_BIT_NV;

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::COMPUTE_PASS)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: IMAGE_SAMPLE -> STORAGE_READ_WRITE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					oldLayout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
					newLayout = RHI::ImageLayout::GENERAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
				}
				else
				{
					if (consumeHistory[right].kind != consumeHistory[left].kind)
					{
						SE_CORE_ERROR("RDG :: unknown access switch");
					}
					i_minus = i;
					continue;
				}

				MemScope<RHI::IImageMemoryBarrier> image_memory_barrier = factory->createImageMemoryBarrier({
					getTexture(), //ITexture* image;
					RHI::ImageSubresourceRange{
						(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::COLOR_BIT,
						0,
						1,
						0,
						1
					},//ImageSubresourceRange subresourceRange;
					srcAccessFlags, //AccessFlags srcAccessMask;
					dstAccessFlags, //AccessFlags dstAccessMask;
					oldLayout, // old Layout
					newLayout // new Layout
				});

				MemScope<RHI::IBarrier> attach_read_barrier = factory->createBarrier({
					srcStageMask,//srcStageMask
					dstStageMask,//dstStageMask
					0,
					{},
					{},
					{image_memory_barrier.get()}
				});

				BarrierHandle barrier_handle = rg->barrierPool.registBarrier(std::move(attach_read_barrier));
				rg->getPassNode(consumeHistory[i].pass)->barriers.emplace_back(barrier_handle);
				i_minus = i;
			}
		}

		if (!hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
		{
			if (consumeHistory.size() > 0)
			{
				RHI::ImageLayout first_layout = RHI::ImageLayout::UNDEFINED;
				unsigned int idx = consumeHistory.size() - 1;
				while (consumeHistory[idx].kind > ConsumeKind::SCOPE) idx--;
				switch (consumeHistory[idx].kind)
				{
				case ConsumeKind::RENDER_TARGET:
					first_layout = RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
					break;
				case ConsumeKind::IMAGE_STORAGE_READ_WRITE:
				case ConsumeKind::INDIRECT_DRAW:
					first_layout = RHI::ImageLayout::GENERAL;
					break;
				case ConsumeKind::IMAGE_SAMPLE:
					first_layout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
					break;
				case ConsumeKind::COPY_SRC:
				case ConsumeKind::COPY_DST:
				case ConsumeKind::BUFFER_READ_WRITE:
				default: SE_CORE_ERROR("RDG :: Color Buffer Undefined initial usage!");
					break;
				}
				getTexture()->transitionImageLayout(RHI::ImageLayout::UNDEFINED, first_layout);
			}

		}
	}
}