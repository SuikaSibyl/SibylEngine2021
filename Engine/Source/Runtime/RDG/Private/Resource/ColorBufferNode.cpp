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

		if (format == RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT)
		{
			hasDepth = true;
			hasStencil = true;
			usages |= (uint32_t)RHI::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT_BIT;
		}
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
		createdBarriers.clear();
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
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> RENDER_TARGET, STORAGE_READ_WRITE is not consumed by a compute pass!");

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::RASTER_PASS_SCOPE)
						dstStageMask = (!hasDepth) ? (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT :
						(uint32_t)RHI::PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT | (uint32_t)RHI::PipelineStageFlagBits::LATE_FRAGMENT_TESTS_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> RENDER_TARGET, RENDER_TARGET is not consumed by a raster pass!");

					oldLayout = RHI::ImageLayout::GENERAL;
					newLayout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::MEMORY_READ_BIT;
					dstAccessFlags = (!hasDepth) ? (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT : 
						(uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT;
				}
				// RENDER_TARGET -> STORE_READ_WRITE
				else if (consumeHistory[left].kind == ConsumeKind::RENDER_TARGET && consumeHistory[right].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE)
				{
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::RASTER_PASS_SCOPE)
						srcStageMask = (!hasDepth) ? (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT :
						(uint32_t)RHI::PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT | (uint32_t)RHI::PipelineStageFlagBits::LATE_FRAGMENT_TESTS_BIT;
					else SE_CORE_ERROR("RDG :: RENDER_TARGET -> STORAGE_READ_WRITE, RENDER_TARGET is not consumed by a raster pass!");

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: RENDER_TARGET -> STORAGE_READ_WRITE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					oldLayout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;
					newLayout = RHI::ImageLayout::GENERAL;

					srcAccessFlags = (!hasDepth) ? (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT : (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
				}
				// STORE_READ_WRITE -> STORE_READ_WRITE
				else if (consumeHistory[left].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE && consumeHistory[right].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE)
				{
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> STORAGE_READ_WRITE, RENDER_TARGET is not consumed by a compute pass!");

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
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
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> IMAGE_SAMPLE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::RASTER_MATERIAL_SCOPE)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT
						| (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;
					else if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::EXTERNAL_ACCESS_PASS)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;

					oldLayout = RHI::ImageLayout::GENERAL;
					newLayout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
				}
				// IMAGE_SAMPLE -> STORE_READ_WRITE
				else if (consumeHistory[left].kind == ConsumeKind::IMAGE_SAMPLE && consumeHistory[right].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE)
				{
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::RASTER_MATERIAL_SCOPE)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT
									 | (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;
									 //| (uint32_t)RHI::PipelineStageFlagBits::MESH_SHADER_BIT_NV;
					else if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::EXTERNAL_ACCESS_PASS)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: IMAGE_SAMPLE -> STORAGE_READ_WRITE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					oldLayout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
					newLayout = RHI::ImageLayout::GENERAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
				}
				// RENDER_TARGET -> IMAGE_SAMPLE
				else if (consumeHistory[left].kind == ConsumeKind::RENDER_TARGET && consumeHistory[right].kind == ConsumeKind::IMAGE_SAMPLE)
				{
					if (rg->getPassNode(consumeHistory[left].pass)->type == NodeDetailedType::RASTER_PASS_SCOPE)
						srcStageMask = (!hasDepth) ? (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT :
						(uint32_t)RHI::PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT | (uint32_t)RHI::PipelineStageFlagBits::LATE_FRAGMENT_TESTS_BIT;
					else SE_CORE_ERROR("RDG :: RENDER_TARGET -> IMAGE_SAMPLE, RENDER_TARGET is not consumed by a raster pass!");

					dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT
									| (uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT
									| (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;

					oldLayout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;
					newLayout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

					srcAccessFlags = (!hasDepth) ? (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT : (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
				}
				// IMAGE_SAMPLE -> RENDER_TARGET
				else if (consumeHistory[left].kind == ConsumeKind::IMAGE_SAMPLE && consumeHistory[right].kind == ConsumeKind::RENDER_TARGET)
				{
					srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT
						| (uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT
						| (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;

					if (rg->getPassNode(consumeHistory[right].pass)->type == NodeDetailedType::RASTER_PASS_SCOPE)
						dstStageMask = (!hasDepth) ? (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT :
						(uint32_t)RHI::PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT | (uint32_t)RHI::PipelineStageFlagBits::LATE_FRAGMENT_TESTS_BIT;
					else SE_CORE_ERROR("RDG :: IMAGE_SAMPLE -> RENDER_TARGET, RENDER_TARGET is not consumed by a raster pass!");

					oldLayout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
					newLayout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
					dstAccessFlags = (!hasDepth) ? (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT : 
						(uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT;
				}
				// RENDER_TARGET -> RENDER_TARGET
				else if (consumeHistory[left].kind == ConsumeKind::RENDER_TARGET && consumeHistory[right].kind == ConsumeKind::RENDER_TARGET)
				{
					srcStageMask = (!hasDepth) ? (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT :
						(uint32_t)RHI::PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT | (uint32_t)RHI::PipelineStageFlagBits::LATE_FRAGMENT_TESTS_BIT;

					dstStageMask = (!hasDepth) ? (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT : 
						(uint32_t)RHI::PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT | (uint32_t)RHI::PipelineStageFlagBits::LATE_FRAGMENT_TESTS_BIT;

					oldLayout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;
					newLayout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;

					srcAccessFlags = (!hasDepth) ? (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT : (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
					dstAccessFlags = (!hasDepth) ? (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT : (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_READ_BIT;
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
						(!hasDepth) ? 
						(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::COLOR_BIT: 
						(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::DEPTH_BIT
						| (RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::STENCIL_BIT,
						0,
						getTexture()->getDescription().mipLevels,
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
				createdBarriers.emplace_back(i, barrier_handle);
				rg->getPassNode(consumeHistory[i].pass)->barriers.emplace_back(barrier_handle);
				i_minus = i;
			}
		}

		if (!hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
		{
			if (consumeHistory.size() > 0)
			{
				unsigned int idx = consumeHistory.size() - 1;
				while (consumeHistory[idx].kind > ConsumeKind::SCOPE) idx--;
				switch (consumeHistory[idx].kind)
				{
				case ConsumeKind::RENDER_TARGET:
					first_layout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;
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
				SE_CORE_DEBUG("LAYOUT {0}", (uint32_t)first_layout);
				getTexture()->transitionImageLayout(RHI::ImageLayout::UNDEFINED, first_layout);
			}

		}
	}
}