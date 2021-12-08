#include "SFileSystem.h"

#include <ctime>
#include <chrono>

namespace SIByL
{
	namespace File
	{
		TimeStamp FileSystem::GetLastWriteTime(const std::filesystem::path& path)
		{
			const auto ftime = std::filesystem::last_write_time(path);
			const auto ticks = ftime.time_since_epoch().count() - __std_fs_file_time_epoch_adjustment;
			const auto tp = std::chrono::system_clock::time_point(std::chrono::system_clock::time_point::duration(ticks));
			const auto tt = std::chrono::system_clock::time_point::clock::to_time_t(tp);

			return tt;
		}

	}
}