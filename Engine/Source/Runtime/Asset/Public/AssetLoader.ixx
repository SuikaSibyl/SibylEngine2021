module;
#include <string>
#include <filesystem>
export module Asset.DedicatedLoader;
import Core.Log;
import Core.File;
import Core.Hash;
import Asset.Asset;
import Asset.RuntimeAssetManager;
import RHI.IFactory;

namespace SIByL::Asset
{
	export struct DedicatedLoader
	{
		DedicatedLoader(RHI::IResourceFactory* factory, RuntimeAssetManager* manager) 
			:resourceFactory(factory), runtimeManager(manager) {}
		virtual auto loadFromFile(std::filesystem::path path) noexcept -> void = 0;
		virtual auto loadFromCache(uint64_t const& path) noexcept -> void = 0;
		virtual auto saveAsCache(uint64_t const& path) noexcept -> void = 0;
		
		virtual auto fromGUID(GUID const& guid) noexcept -> void;

		RHI::IResourceFactory* resourceFactory;
		RuntimeAssetManager* runtimeManager;
	};

	auto DedicatedLoader::fromGUID(GUID const& guid) noexcept -> void
	{
		ResourceItem* item = runtimeManager->findAsset(guid);
		if (item)
		{
			AssetLoader* loader = runtimeManager->getAssetLoader();
			bool file_exist = loader->fileExists(item->path);
			bool cache_exist = loader->fileExists(std::string("../cache/") + std::to_string(item->cacheID));
			TimeStamp last_write = 0;
			if(file_exist) last_write = loader->getFileLastWriteTime(item->path);

			// If origin file not exists || have been cached
			if (cache_exist && (!file_exist || last_write == item->cachedTime))
			{
				loadFromCache(item->cacheID);
			}
			else if (file_exist)
			{
				loadFromFile(item->path);
				if (item->cacheID == 0) item->cacheID = Hash::path2hash(item->path);
				saveAsCache(item->cacheID);
				item->cachedTime = last_write;
			}
			else
			{
				SE_CORE_ERROR("Asset :: Asset File not Found, Neither origin Nor Cache.");
			}
		}
		else
		{
			SE_CORE_ERROR("Asset :: Asset GUID Not Found, Input GUID : {0}", guid);
		}
	}

}
