#include "SFileSystem.h"

#include <ctime>
#include <chrono>

using namespace std::chrono;

namespace SIByL
{
	namespace File
	{
		TimeStamp FileSystem::GetLastWriteTime(const std::filesystem::path& path)
		{
			const auto ftime = std::filesystem::last_write_time(path);
			const auto ticks = ftime.time_since_epoch().count() - 0x19DB1DED53E8000LL;
			const auto tp = std::chrono::system_clock::time_point(std::chrono::system_clock::time_point::duration(ticks));
			const auto tt = std::chrono::system_clock::time_point::clock::to_time_t(tp);

			return tt;
		}

	}
}