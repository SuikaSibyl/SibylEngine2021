export module Asset.Texture;
import Core.MemoryManager;
import Asset.Asset;
import RHI.ITexture;
import RHI.ITextureView;

namespace SIByL::Asset
{
	export struct Texture :public Asset
	{
		MemScope<RHI::ITexture> texture;
		MemScope<RHI::ITextureView> view;
	};
}