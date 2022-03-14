module;
#include <cstdint>
#include <vector>
export module GFX.RDG.TextureBufferNode;
import Core.BitFlag;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.IFactory;
import GFX.RDG.Common;

namespace SIByL::GFX::RDG
{
	export struct TextureBufferNode :public ResourceNode
	{
		auto getTexture() noexcept -> RHI::ITexture*
		{
			if (!hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
			{
				return texture.get();
			}
			return ext_texture;
		}

		auto getTextureView() noexcept -> RHI::ITextureView*
		{
			if (!hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
			{
				return view.get();
			}
			return ext_view;
		}

		RHI::ITexture* ext_texture;
		RHI::ITextureView* ext_view;

		MemScope<RHI::ITexture> texture;
		MemScope<RHI::ITextureView> view;
	};
}