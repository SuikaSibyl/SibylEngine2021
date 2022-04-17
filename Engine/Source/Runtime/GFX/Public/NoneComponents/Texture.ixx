export module GFX.Texture;
import Core.MemoryManager;
import Asset.Asset;
import RHI.ITexture;
import RHI.ITextureView;
import Asset.Texture;
import Asset.AssetLayer;

namespace SIByL::GFX
{
	export struct Texture
	{
		static auto query(Asset::GUID guid, Asset::AssetLayer* layer) noexcept -> Texture;

		Asset::GUID guid;
		RHI::ITexture* texture;
		RHI::ITextureView* view;
	};

	auto Texture::query(Asset::GUID guid, Asset::AssetLayer* layer) noexcept -> Texture
	{
		Texture texture;
		Asset::Texture* asset_texture = layer->texture(guid);
		texture.guid = guid;
		texture.texture = asset_texture->texture.get();
		texture.view = asset_texture->view.get();
		return texture;
	}

}