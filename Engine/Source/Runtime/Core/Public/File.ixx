module;
#include <iostream>
#include <vector>
#include <filesystem>
export module Core.File;
import Core.SObject;
import Core.MemoryManager;
import Core.Buffer;

namespace SIByL
{
	inline namespace Core
	{
		export class AssetLoader :public SObject
		{
		public:
			using AssetFilePtr = void*;

			AssetLoader() = default;
			AssetLoader(std::vector<std::filesystem::path> const& pathes);
			virtual ~AssetLoader() = default;

			auto addSearchPath(std::filesystem::path const& path) noexcept -> void;
			auto removeSearchPath(std::filesystem::path const& path) noexcept -> void;

			auto findRelativePath(std::filesystem::path const& relative) noexcept -> std::filesystem::path;
			auto getFileLastWriteTime(std::filesystem::path const& relative) noexcept -> uint64_t;

			auto fileExists(std::filesystem::path filePath) noexcept -> bool;
			auto openFile(std::filesystem::path const& name) noexcept -> AssetFilePtr;
			auto openFileWB(std::filesystem::path const& name) noexcept -> AssetFilePtr;
			auto closeFile(AssetFilePtr& fp) noexcept -> void;
			auto getSize(AssetFilePtr& fp) noexcept -> size_t;
			auto syncReadAll(AssetFilePtr& fp, Buffer& buf) -> void;
			auto readBuffer(AssetFilePtr& fp, Buffer const& buf) -> void;
			auto syncReadAll(std::filesystem::path const& name, Buffer& buf) -> void;
			auto syncWriteAll(std::filesystem::path const& name, Buffer& buf) -> void;
			auto writeBuffer(AssetFilePtr& fp, Buffer const& buf) -> void;

		private:
			std::vector<std::filesystem::path> searchPathes;
		};
	}
}