module;
#include <cstdint>
#include <vector>
export module GFX.RDG.ColorBufferNode;
import GFX.RDG.Common;
import Core.BitFlag;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.IFactory;
import GFX.RDG.TextureBufferNode;

namespace SIByL::GFX::RDG
{
	export struct ColorBufferNode :public TextureBufferNode
	{
		ColorBufferNode() { type = NodeDetailedType::COLOR_TEXTURE; }
		ColorBufferNode(RHI::ResourceFormat format, float const& rel_width, float const& rel_height);
		
		RHI::ImageLayout first_layout = RHI::ImageLayout::UNDEFINED;

		virtual auto devirtualize(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		virtual auto rereference(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;

		virtual auto onBuild(void* graph, RHI::IResourceFactory* factory) noexcept -> void override;
		uint32_t mipLevels = 1;

		bool hasDepth = false;
		bool hasStencil = false;
	};
}