#pragma once

#include <filesystem>

namespace SIByL
{
	namespace File
	{
		using TimeStamp = uint64_t;

		class FileSystem
		{
		public:
			static TimeStamp GetLastWriteTime(const std::filesystem::path& path);

		};
	}
}