module;
#include <cstdint>
#include <vector>
export module GFX.RDG.TextureBufferNode;
import GFX.RDG.ResourceNode;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.ITexture;
import RHI.ITextureView;
import RHI.IFactory;

namespace SIByL::GFX::RDG
{
	export struct TextureBufferNode :public ResourceNode
	{
		MemScope<RHI::ITexture> depthTexture;
		MemScope<RHI::ITextureView> depthView;
	};
}