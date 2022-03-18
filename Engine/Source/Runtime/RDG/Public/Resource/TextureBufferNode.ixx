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
		auto getTexture() noexcept -> RHI::ITexture*;
		auto getTextureView() noexcept -> RHI::ITextureView*;
		auto resetExternal(RHI::ITexture* t, RHI::ITextureView* tv) noexcept -> void;

		float relWidth = 0, relHeight = 0;
		RHI::ResourceFormat format;
		RHI::ImageUsageFlags usages = 0;

		TolerantPtr<RHI::ITexture> texture;
		TolerantPtr<RHI::ITextureView> textureView;
	};

	// Impl
	// --------------------------------------

	auto TextureBufferNode::getTexture() noexcept -> RHI::ITexture*
	{
		if (!hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
		{
			return texture.scope.get();
		}
		return texture.ref;
	}

	auto TextureBufferNode::getTextureView() noexcept -> RHI::ITextureView*
	{
		if (!hasBit(attributes, NodeAttrbutesFlagBits::PLACEHOLDER))
		{
			return textureView.scope.get();
		}
		return textureView.ref;
	}

	auto TextureBufferNode::resetExternal(RHI::ITexture* t, RHI::ITextureView* tv) noexcept -> void
	{
		texture.ref = t;
		textureView.ref = tv;
	}
}