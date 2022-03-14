module;
#include <cstdint>
#include <vector>
export module GFX.RDG.DepthBufferNode;
import GFX.RDG.Common;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.IFactory;
import GFX.RDG.RenderGraph;
import GFX.RDG.TextureBufferNode;

namespace SIByL::GFX::RDG
{
	export struct DepthBufferNode :public TextureBufferNode
	{
		DepthBufferNode(float const& rel_width, float const& rel_height);
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;

		float relWidth, relHeight;
	};

	DepthBufferNode::DepthBufferNode(float const& rel_width, float const& rel_height)
		: relWidth(rel_width)
		, relHeight(rel_height)
	{}

	auto DepthBufferNode::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		onReDatum(graph, factory);
	}

	auto DepthBufferNode::onReDatum(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)graph;

		texture = factory->createTexture(
			{
			RHI::ResourceType::Texture2D, //ResourceType type;
			RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT, //ResourceFormat format;
			RHI::ImageTiling::OPTIMAL, //ImageTiling tiling;
			(uint32_t)RHI::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT_BIT, //ImageUsageFlags usages;
			RHI::BufferShareMode::EXCLUSIVE, //BufferShareMode shareMode;
			RHI::SampleCount::COUNT_1_BIT, //SampleCount sampleCount;
			RHI::ImageLayout::UNDEFINED, //ImageLayout layout;
			(uint32_t)(render_graph->getDatumWidth() * relWidth), //uint32_t width;
			(uint32_t)(render_graph->getDatumHeight() * relHeight) //uint32_t height;
			});

		view = factory->createTextureView(texture.get());
	}
}