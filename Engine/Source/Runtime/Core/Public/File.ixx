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
			virtual ~AssetLoader() = default;

			auto addSearchPath(std::filesystem::path const& path) noexcept -> void;
			auto removeSearchPath(std::filesystem::path const& path) noexcept -> void;

			auto fileExists(std::filesystem::path filePath) noexcept -> bool;
			auto openFile(std::filesystem::path const& name) noexcept -> AssetFilePtr;
			auto closeFile(AssetFilePtr& fp) noexcept -> void;
			auto getSize(AssetFilePtr& fp) noexcept -> size_t;
			auto syncReadAll(AssetFilePtr& fp, Buffer& buf) -> void;
			auto syncReadAll(std::filesystem::path const& name, Buffer& buf) -> void;

		private:
			std::vector<std::filesystem::path> searchPathes;
		};
	}
}