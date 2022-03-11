module;
#include <cstdint>
#include <vector>
export module GFX.RDG.ColorBufferNode;
import GFX.RDG.ResourceNode;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.IFactory;
import GFX.RDG.RenderGraph;

namespace SIByL::GFX::RDG
{
	export struct ColorBufferNode :public ResourceNode
	{
		ColorBufferNode(RHI::ResourceFormat format, float const& rel_width, float const& rel_height);
		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;

		float relWidth, relHeight;
		RHI::ResourceFormat format;
		RHI::ImageUsageFlags usages;
		MemScope<RHI::ITexture> colorTexture;
		MemScope<RHI::ITextureView> colorView;
	};

	ColorBufferNode::ColorBufferNode(RHI::ResourceFormat format, float const& rel_width, float const& rel_height)
		: relWidth(rel_width)
		, relHeight(rel_height)
		, format(format)
	{}

	auto ColorBufferNode::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)graph;

		colorTexture = factory->createTexture(
			{
			RHI::ResourceType::Texture2D, //ResourceType type;
			format, //ResourceFormat format;
			RHI::ImageTiling::OPTIMAL, //ImageTiling tiling;
			(uint32_t)RHI::ImageUsageFlagBits::DEPTH_STENCIL_ATTACHMENT_BIT, //ImageUsageFlags usages;
			RHI::BufferShareMode::EXCLUSIVE, //BufferShareMode shareMode;
			RHI::SampleCount::COUNT_1_BIT, //SampleCount sampleCount;
			RHI::ImageLayout::UNDEFINED, //ImageLayout layout;
			(uint32_t)(render_graph->getDatumWidth() * relWidth), //uint32_t width;
			(uint32_t)(render_graph->getDatumHeight() * relHeight) //uint32_t height;
			});

		colorView = factory->createTextureView(colorTexture.get());
	}

}