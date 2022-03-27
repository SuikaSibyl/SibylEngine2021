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

		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto rereference(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;

		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
	};

	// Impl
	// ---------------------

	DepthBufferNode::DepthBufferNode(float const& rel_width, float const& rel_height)
	{ 
		relWidth = rel_width;
		relHeight = rel_height;
		format = RHI::ResourceFormat::FORMAT_D24_UNORM_S8_UINT;
		type = NodeDetailedType::DEPTH_TEXTURE; 
	}

	auto DepthBufferNode::devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		RenderGraph* render_graph = (RenderGraph*)graph;

		texture.scope = factory->createTexture(
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

		textureView.scope = factory->createTextureView(getTexture());
	}

	auto DepthBufferNode::rereference(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{
		devirtualize(graph, factory);
	}

	auto DepthBufferNode::onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void
	{}
}