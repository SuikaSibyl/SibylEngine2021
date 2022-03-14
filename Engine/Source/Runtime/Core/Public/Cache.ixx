module;
#include <filesystem>
export module Core.Cache;
import Core.File;
import Core.Buffer;

namespace SIByL
{
	inline namespace Core
	{
		// Assets will be managed with unique identifier & timestamp
		// Two parts are likely to exist for an asset:
		// 1. source file
		// 2. cache file
		// unique identifier is corresponded to relative path
		// when we get a path try to find the source file & cache file
		// simple rules are applied here:

		// 1. case "no source file":
		//    load cache
		// 2. case "source file timestamp == cache's inner timestamp":
		//    load cache
		// 3. else
		//    load source file & save to cache

		export struct EmptyHeader
		{};

		export class CacheBrain
		{
		public:
			CacheBrain();

			static auto instance() noexcept -> CacheBrain*;

			template <class Header>
			auto saveCache(uint64_t identifier, Header const& header, Buffer** buffers, uint32_t buffers_count, uint64_t const& timeStamp) noexcept -> void;

			template <class Header>
			auto loadCache(uint64_t const& uid, Header& header, Buffer** buffers) noexcept -> void;
			
		private:
			AssetLoader sourceLoader;
			AssetLoader cacheLoader;
		};

		// Cache content
		// 
		// 1. CacheMetaData
		// 2. HeaderData
		// 3. BufferMetaData 0
		//    BufferMetaData 1
		//    ...
		// 4. BufferData 0
		//    BufferData 1
		//    ...

		struct CacheMetaData
		{
			uint64_t timeStamp;
			uint32_t headerLength;
			uint32_t bufferCount;
		};

		struct BufferMetaData
		{
			uint64_t bufferLength;
			uint64_t bufferStride;
		};

		template <class Header>
		auto CacheBrain::saveCache(uint64_t id, Header const& header, Buffer** buffers, uint32_t buffers_count, uint64_t const& timeStamp) noexcept -> void
		{
			// open output
			std::filesystem::path path = std::to_string(id);
			AssetLoader::AssetFilePtr file_ptr = cacheLoader.openFileWB(path);
			// write CacheMetaData
			CacheMetaData metadata{ timeStamp, sizeof(Header), buffers_count };
			cacheLoader.writeBuffer(file_ptr, Buffer((void*)&metadata, sizeof(metadata), 1));
			// wrtie HeaderData
			cacheLoader.writeBuffer(file_ptr, Buffer((void*)&header, sizeof(Header), 1));
			// write BufferMetaDatas
			std::vector<BufferMetaData> buffer_metadatas(buffers_count);
			for (int i = 0; i < buffers_count; i++)
				buffer_metadatas[i] = { buffers[i]->getSize(), buffers[i]->getStride() };
			cacheLoader.writeBuffer(file_ptr, Buffer((void*)buffer_metadatas.data(), sizeof(BufferMetaData) * buffers_count, 1));
			// write BufferDatas
			for (int i = 0; i < buffers_count; i++)
				cacheLoader.writeBuffer(file_ptr, *(buffers[i]));
			cacheLoader.closeFile(file_ptr);
		}

		template <class Header>
		auto CacheBrain::loadCache(uint64_t const& uid, Header& header, Buffer** buffers) noexcept -> void
		{
			// open output
			std::filesystem::path path = std::to_string(uid);
			AssetLoader::AssetFilePtr file_ptr = cacheLoader.openFile(path);
			// read CacheMetaData
			CacheMetaData metadata{};
			Buffer metadata_proxy((void*)&metadata, sizeof(metadata), 1);
			cacheLoader.readBuffer(file_ptr, metadata_proxy);
			// read HeaderData
			Buffer header_proxy((void*)&header, sizeof(Header), 1);
			cacheLoader.readBuffer(file_ptr, header_proxy);
			// read BufferMetaDatas
			std::vector<BufferMetaData> buffer_metadatas(metadata.bufferCount);
			Buffer buffer_metadata_proxy((void*)buffer_metadatas.data(), sizeof(BufferMetaData) * metadata.bufferCount, 1);
			cacheLoader.readBuffer(file_ptr, buffer_metadata_proxy);
			// read BufferDatas
			for (int i = 0; i < metadata.bufferCount; i++)
			{
				*buffers[i] = std::move(Buffer{ buffer_metadatas[i].bufferLength, buffer_metadatas[i].bufferStride });
				cacheLoader.readBuffer(file_ptr, *buffers[i]);
			}
			cacheLoader.closeFile(file_ptr);
		}
	}
}