module;
#include <cstdint>
#include <vector>
#include <algorithm>
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
import RHI.ILogicalDevice;
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
		rasterStages = factory->getLogicalDevice()->getRasterStageMask();
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
				relWidth > 0 ? (uint32_t)(render_graph->getDatumWidth() * relWidth) : (uint32_t)(-relWidth), //uint32_t width;
				relHeight > 0 ? (uint32_t)(render_graph->getDatumHeight() * relHeight) : (uint32_t)(-relHeight), //uint32_t height;
				mipLevels // uint32_t mip levels
				});
			textureView.scope = factory->createTextureView(getTexture(), usages);
		}
	}

	auto ColorBufferNode::rereference(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		devirtualize(graph, factory);
	}

	auto subResourceFilter(
		std::vector<ConsumeHistory> const& consumeHistory, int& neighbor_idx,
		uint32_t subpara0, uint32_t subpara1, uint32_t subpara2, uint32_t subpara3) noexcept -> std::vector<int>
	{
		std::vector<int> result;
		for (int i = 0; i < consumeHistory.size(); i++)
		{
			ConsumeHistory const& item = consumeHistory[i];
			if (consumeHistory[i].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN)
			{
				result.emplace_back(i);
			}
			else if (consumeHistory[i].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_END)
			{
				if (consumeHistory[result.back()].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN)
					result.pop_back();
				else
					result.emplace_back(i);
			}
			else if (item.subResourcePara_0 <= subpara0 && item.subResourcePara_0 + item.subResourcePara_1 >= subpara0 + subpara1)
				result.emplace_back(i);
			if (i == neighbor_idx)
			{
				neighbor_idx = result.size() - 1;
			}
		}
		return result;
	}

	struct SubResourceRange
	{
		uint32_t start;
		uint32_t end;
		ConsumeKind lastConsume;
		uint32_t lastConsumeDepth = 0;
	};

	struct ResourceDivision
	{
		auto insertNewRange(SubResourceRange const& range) noexcept -> void
		{
			points.emplace_back(range.start);
			points.emplace_back(range.end);
			ranges.emplace_back(range);
		}

		auto build() noexcept -> void
		{
			// get all points
			std::sort(points.begin(), points.end());
			points.erase(std::unique(points.begin(), points.end()), points.end());
			// get all ranges
			for (int i = 0; i < points.size() - 1; i++)
			{
				cut_ranges.emplace_back(points[i], points[i + 1]);
			}
			// 
			for (auto& cut_range : cut_ranges)
			{
				for (auto& range : ranges)
				{
					if (range.start <= cut_range.start && range.end >= cut_range.end)
					{
						if (range.lastConsumeDepth >= cut_range.lastConsumeDepth &&
							range.lastConsume < ConsumeKind::SCOPE)
						{
							cut_range.lastConsume = range.lastConsume;
							cut_range.lastConsumeDepth = range.lastConsumeDepth;
						}
					}
				}
			}
		}
		std::vector<int> points;
		std::vector<SubResourceRange> cut_ranges;
		std::vector<SubResourceRange> ranges;
	};

	auto ColorBufferNode::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		
		createdBarriers.clear();
		RenderGraph* rg = (RenderGraph*)graph;

		// Do Resource Division
		ResourceDivision resourceDivision;
		unsigned int history_size = consumeHistory.size();
		if (history_size == 0) return;
		// Do Resource Division
		for (int i = 0; i < history_size; i++)
		{
			if (consumeHistory[i].subResourcePara_1 == (uint32_t)(-1))
				consumeHistory[i].subResourcePara_1 = getTexture()->getDescription().mipLevels;
			int valid_idx = i;
			std::vector<int> valid_neighbors = subResourceFilter(
				consumeHistory, valid_idx,
				consumeHistory[i].subResourcePara_0,
				consumeHistory[i].subResourcePara_1,
				consumeHistory[i].subResourcePara_2,
				consumeHistory[i].subResourcePara_3
			);
			resourceDivision.insertNewRange({
				consumeHistory[i].subResourcePara_0,
				consumeHistory[i].subResourcePara_0 + consumeHistory[i].subResourcePara_1,
				consumeHistory[valid_neighbors.back()].kind,
				(uint32_t)valid_neighbors.back()
				});
		}
		resourceDivision.build();


		if (consumeHistory.size() > 1)
		{
			if (consumeHistory[0].kind >= ConsumeKind::SCOPE && consumeHistory.size() == 3) return;

			unsigned int i_minus = history_size - 1;

			for (int i = 0; i < history_size; i++)
			{			
				int valid_idx = i;
				std::vector<int> valid_neighbors = subResourceFilter(
					consumeHistory, valid_idx,
					consumeHistory[i].subResourcePara_0,
					consumeHistory[i].subResourcePara_1,
					consumeHistory[i].subResourcePara_2,
					consumeHistory[i].subResourcePara_3
				);

				int right = valid_idx;
				int left = valid_idx - 1 < 0 ? valid_idx - 1 + valid_neighbors.size() : valid_idx - 1;

				// - A - BEGIN - B -
				//   |     | 
				// we create a A-B barrier in BEGIN
				if (consumeHistory[valid_neighbors[right]].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN)
				{
					right = right + 1;
				}
				// - BEGIN - A - B - C - END -
				//     |     |
				// we create a C-A barrier in A
				if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN)
				{
					while (consumeHistory[valid_neighbors[left + 1]].kind != ConsumeKind::MULTI_DISPATCH_SCOPE_END)
					{
						left++;
					}
				}
				// - A - BEGIN - B - C - END - D
				//                        |    |
				// we create a C-D barrier in D
				if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_END)
				{
					left--;
				}
				// - A - BEGIN - B - C - END - D
				//                   |    |   
				// we create a A-D barrier in End ( used only when 0 dispatch happens)
				if (consumeHistory[valid_neighbors[right]].kind == ConsumeKind::MULTI_DISPATCH_SCOPE_END)
				{
					while (consumeHistory[valid_neighbors[left + 1]].kind != ConsumeKind::MULTI_DISPATCH_SCOPE_BEGIN)
					{
						left--;
					}
					right = right + 1;
				}

				if (left < 0) left += valid_neighbors.size();
				if (right >= valid_neighbors.size()) right -= valid_neighbors.size();

				while (consumeHistory[valid_neighbors[left]].kind >= ConsumeKind::SCOPE) left--;
				while (consumeHistory[valid_neighbors[right]].kind >= ConsumeKind::SCOPE) right++;

				RHI::PipelineStageFlags srcStageMask = 0;
				RHI::PipelineStageFlags dstStageMask = 0;
				RHI::ImageLayout oldLayout = RHI::ImageLayout::UNDEFINED;
				RHI::ImageLayout newLayout = RHI::ImageLayout::UNDEFINED;
				RHI::AccessFlags srcAccessFlags = 0;
				RHI::AccessFlags dstAccessFlags = 0;

				// STORE_READ_WRITE -> RENDER_TARGET
				if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE && consumeHistory[valid_neighbors[right]].kind == ConsumeKind::RENDER_TARGET)
				{
					if (rg->getPassNode(consumeHistory[valid_neighbors[left]].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> RENDER_TARGET, STORAGE_READ_WRITE is not consumed by a compute pass!");

					if (rg->getPassNode(consumeHistory[valid_neighbors[right]].pass)->type == NodeDetailedType::RASTER_PASS_SCOPE)
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
				else if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::RENDER_TARGET && consumeHistory[valid_neighbors[right]].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE)
				{
					if (rg->getPassNode(consumeHistory[valid_neighbors[left]].pass)->type == NodeDetailedType::RASTER_PASS_SCOPE)
						srcStageMask = (!hasDepth) ? (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT :
						(uint32_t)RHI::PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT | (uint32_t)RHI::PipelineStageFlagBits::LATE_FRAGMENT_TESTS_BIT;
					else SE_CORE_ERROR("RDG :: RENDER_TARGET -> STORAGE_READ_WRITE, RENDER_TARGET is not consumed by a raster pass!");

					if (rg->getPassNode(consumeHistory[valid_neighbors[right]].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: RENDER_TARGET -> STORAGE_READ_WRITE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					oldLayout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;
					newLayout = RHI::ImageLayout::GENERAL;

					srcAccessFlags = (!hasDepth) ? (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT : (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
				}
				// STORE_READ_WRITE -> STORE_READ_WRITE
				else if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE && consumeHistory[valid_neighbors[right]].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE)
				{
					if (rg->getPassNode(consumeHistory[valid_neighbors[left]].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> STORAGE_READ_WRITE, RENDER_TARGET is not consumed by a compute pass!");

					if (rg->getPassNode(consumeHistory[valid_neighbors[right]].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> STORAGE_READ_WRITE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					oldLayout = RHI::ImageLayout::GENERAL;
					newLayout = RHI::ImageLayout::GENERAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT | (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
				}
				// STORE_READ_WRITE -> IMAGE_SAMPLE
				else if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE && consumeHistory[valid_neighbors[right]].kind == ConsumeKind::IMAGE_SAMPLE)
				{
					if (rg->getPassNode(consumeHistory[valid_neighbors[left]].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: STORAGE_READ_WRITE -> IMAGE_SAMPLE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					if (rg->getPassNode(consumeHistory[valid_neighbors[right]].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else if (rg->getPassNode(consumeHistory[valid_neighbors[right]].pass)->type == NodeDetailedType::RASTER_MATERIAL_SCOPE)
						dstStageMask = rasterStages;
					else if (rg->getPassNode(consumeHistory[valid_neighbors[right]].pass)->type == NodeDetailedType::EXTERNAL_ACCESS_PASS)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;

					oldLayout = RHI::ImageLayout::GENERAL;
					newLayout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
				}
				// IMAGE_SAMPLE -> STORE_READ_WRITE
				else if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::IMAGE_SAMPLE && consumeHistory[valid_neighbors[right]].kind == ConsumeKind::IMAGE_STORAGE_READ_WRITE)
				{
					if (rg->getPassNode(consumeHistory[valid_neighbors[left]].pass)->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else if (rg->getPassNode(consumeHistory[valid_neighbors[left]].pass)->type == NodeDetailedType::RASTER_MATERIAL_SCOPE)
						srcStageMask = rasterStages;
					else if (rg->getPassNode(consumeHistory[valid_neighbors[left]].pass)->type == NodeDetailedType::EXTERNAL_ACCESS_PASS)
						srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;

					auto pass = rg->getPassNode(consumeHistory[valid_neighbors[right]].pass);
					if (pass->type == NodeDetailedType::COMPUTE_MATERIAL_SCOPE)
						dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT;
					else SE_CORE_ERROR("RDG :: IMAGE_SAMPLE -> STORAGE_READ_WRITE, STORAGE_READ_WRITE is not consumed by a compute pass!");

					oldLayout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
					newLayout = RHI::ImageLayout::GENERAL;

					srcAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_WRITE_BIT;
				}
				// RENDER_TARGET -> IMAGE_SAMPLE
				else if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::RENDER_TARGET && consumeHistory[valid_neighbors[right]].kind == ConsumeKind::IMAGE_SAMPLE)
				{
					if (rg->getPassNode(consumeHistory[valid_neighbors[left]].pass)->type == NodeDetailedType::RASTER_PASS_SCOPE)
						srcStageMask = (!hasDepth) ? (uint32_t)RHI::PipelineStageFlagBits::COLOR_ATTACHMENT_OUTPUT_BIT :
						(uint32_t)RHI::PipelineStageFlagBits::EARLY_FRAGMENT_TESTS_BIT | (uint32_t)RHI::PipelineStageFlagBits::LATE_FRAGMENT_TESTS_BIT;
					else SE_CORE_ERROR("RDG :: RENDER_TARGET -> IMAGE_SAMPLE, RENDER_TARGET is not consumed by a raster pass!");

					dstStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT
									| rasterStages;

					oldLayout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;
					newLayout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

					srcAccessFlags = (!hasDepth) ? (uint32_t)RHI::AccessFlagBits::COLOR_ATTACHMENT_WRITE_BIT : (uint32_t)RHI::AccessFlagBits::DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
					dstAccessFlags = (uint32_t)RHI::AccessFlagBits::SHADER_READ_BIT;
				}
				// IMAGE_SAMPLE -> RENDER_TARGET
				else if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::IMAGE_SAMPLE && consumeHistory[valid_neighbors[right]].kind == ConsumeKind::RENDER_TARGET)
				{
					srcStageMask = (uint32_t)RHI::PipelineStageFlagBits::COMPUTE_SHADER_BIT
						| rasterStages;

					if (rg->getPassNode(consumeHistory[valid_neighbors[right]].pass)->type == NodeDetailedType::RASTER_PASS_SCOPE)
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
				else if (consumeHistory[valid_neighbors[left]].kind == ConsumeKind::RENDER_TARGET && consumeHistory[valid_neighbors[right]].kind == ConsumeKind::RENDER_TARGET)
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
					if (consumeHistory[valid_neighbors[right]].kind != consumeHistory[valid_neighbors[left]].kind)
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
						(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::COLOR_BIT :
						(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::DEPTH_BIT
						| (RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::STENCIL_BIT,
						consumeHistory[valid_neighbors[right]].subResourcePara_0,
						consumeHistory[valid_neighbors[right]].subResourcePara_1 == (uint32_t)(-1) ? getTexture()->getDescription().mipLevels : consumeHistory[valid_neighbors[right]].subResourcePara_1,
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

				MemScope<RHI::IBarrier> barrierInit = nullptr;
				MemScope<RHI::IImageMemoryBarrier> init_image_memory_barrier = nullptr;
				if (valid_idx == 0)
				{
					init_image_memory_barrier = std::move(factory->createImageMemoryBarrier({
						getTexture(), //ITexture* image;
						RHI::ImageSubresourceRange{
							(!hasDepth) ?
							(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::COLOR_BIT :
							(RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::DEPTH_BIT
							| (RHI::ImageAspectFlags)RHI::ImageAspectFlagBits::STENCIL_BIT,
							consumeHistory[valid_neighbors[right]].subResourcePara_0,
							consumeHistory[valid_neighbors[right]].subResourcePara_1 == (uint32_t)(-1) ? getTexture()->getDescription().mipLevels : consumeHistory[valid_neighbors[right]].subResourcePara_1,
							0,
							1
						},//ImageSubresourceRange subresourceRange;
						0, //AccessFlags srcAccessMask;
						dstAccessFlags, //AccessFlags dstAccessMask;
						RHI::ImageLayout::UNDEFINED, // old Layout
						newLayout // new Layout
						}));

					barrierInit = std::move(factory->createBarrier({
						(uint32_t)RHI::PipelineStageFlagBits::TOP_OF_PIPE_BIT,//srcStageMask
						dstStageMask,//dstStageMask
						0,
						{},
						{},
						{init_image_memory_barrier.get()}
						}));
				}

				BarrierHandle barrier_handle = rg->barrierPool.registBarrier(std::move(attach_read_barrier), std::move(barrierInit));
				createdBarriers.emplace_back(i, barrier_handle);
				rg->getPassNode(consumeHistory[i].pass)->barriers.emplace_back(barrier_handle);
				i_minus = i;
			}
		}
		else if (consumeHistory.size() == 1)
		{
			if (!hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
			{
				getTexture()->transitionImageLayout(RHI::ImageLayout::UNDEFINED, RHI::ImageLayout::GENERAL, { 0,0,1,0,1 });
			}
		}

		//if (!hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
		//{
		//	if (consumeHistory.size() > 0)
		//	{
		//		for (auto range : resourceDivision.cut_ranges)
		//		{
		//			//unsigned int idx = consumeHistory.size() - 1;
		//			//while (range.lastConsume > ConsumeKind::SCOPE) idx--;
		//			switch (range.lastConsume)
		//			{
		//			case ConsumeKind::RENDER_TARGET:
		//				first_layout = (!hasDepth) ? RHI::ImageLayout::COLOR_ATTACHMENT_OPTIMAL : RHI::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMA;
		//				break;
		//			case ConsumeKind::IMAGE_STORAGE_READ_WRITE:
		//			case ConsumeKind::INDIRECT_DRAW:
		//				first_layout = RHI::ImageLayout::GENERAL;
		//				break;
		//			case ConsumeKind::IMAGE_SAMPLE:
		//				first_layout = RHI::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
		//				break;
		//			case ConsumeKind::COPY_SRC:
		//			case ConsumeKind::COPY_DST:
		//			case ConsumeKind::BUFFER_READ_WRITE:
		//			default: SE_CORE_ERROR("RDG :: Color Buffer Undefined initial usage!");
		//				break;
		//			}
		//			getTexture()->transitionImageLayout(RHI::ImageLayout::UNDEFINED, first_layout, {
		//				0,
		//				range.start,
		//				range.end - range.start,
		//				0,
		//				1,
		//				});

		//		}
		//	}
		//}
	}
}