module;
#include <string>
#include <filesystem>
export module Asset.TextureLoader;
import Core.Buffer;
import Core.Image;
import Core.Cache;
import Core.MemoryManager;
import Asset.Asset;
import Asset.DedicatedLoader;
import Asset.Texture;
import RHI.IFactory;
import RHI.ITexture;
import RHI.ITextureView;


namespace SIByL::Asset
{
	export struct TextureLoader :public DedicatedLoader
	{
		TextureLoader(Texture& texture, RHI::IResourceFactory* factory, RuntimeAssetManager* manager)
			:DedicatedLoader(factory, manager), texture(texture) {}

		virtual auto loadFromFile(std::filesystem::path path) noexcept -> void override;
		virtual auto loadFromCache(uint64_t const& path) noexcept -> void override;
		virtual auto saveAsCache(uint64_t const& path) noexcept -> void override;

		Image image;
		Texture& texture;
	};

	auto TextureLoader::loadFromFile(std::filesystem::path path) noexcept -> void
	{
		image = std::move(Image(path));
		texture.texture = resourceFactory->createTexture(&image);
		texture.view = resourceFactory->createTextureView(texture.texture.get());
	}

	struct TextureHeader
	{
		uint32_t attributes;
		uint32_t width;
		uint32_t height;
		uint32_t channels;
		uint32_t size;
	};

	auto TextureLoader::loadFromCache(uint64_t const& path) noexcept -> void
	{
		//TextureHeader header;
		//Buffer image_proxy(image.data, 0, 1);
		//Buffer* buffers[1] = { &image_proxy };
		//CacheBrain::instance()->loadCache(path, header, buffers);
		//image.attributes = header.attributes;
		//image.width = header.width;
		//image.height = header.height;
		//image.channels = header.channels;
		//image.size = header.size;
	}

	auto TextureLoader::saveAsCache(uint64_t const& path) noexcept -> void
	{
		//TextureHeader header;
		//header.attributes = image.attributes;
		//header.width = image.width;
		//header.height = image.height;
		//header.channels = image.channels;
		//header.size = image.size;
		//Buffer image_proxy(image.data, image.size, 1);
		//Buffer* buffers[1] = { &image_proxy };
		//CacheBrain::instance()->saveCache(path, header, buffers, 1, 0);
	}
}