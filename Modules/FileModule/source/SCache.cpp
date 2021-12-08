#include "SCache.h"
#include <fstream>

namespace SIByL
{
	namespace File
	{
		SCache::SCache(const std::filesystem::path& path)
			:path(path)
		{

		}

		void SCache::Serialize()
		{
			std::ofstream ofstream(path);
			ofstream.write((char*)&version, sizeof(TimeStamp));
			ofstream.write((char*)&bufferCount, sizeof(unsigned int));

			for (int i = 0; i < bufferCount; i++)
			{
				mBuffers[i].Serialize(ofstream);
				mBuffers[i].ReleaseBuffer();
			}
			ofstream.close();
		}

		void SCache::Deserialize()
		{
			std::ifstream ifstream(path);

			ifstream.read((char*)&version, sizeof(TimeStamp));
			ifstream.read((char*)&bufferCount, sizeof(unsigned int));

			mBuffers.resize(bufferCount);
			for (int i = 0; i < bufferCount; i++)
			{
				mBuffers[i].Deserialize(ifstream);
			}

			bufferCount = 0;
			ifstream.close();
		}

		TimeStamp SCache::GetVersion()
		{
			return version;
		}

		void SCache::SetVersion(const TimeStamp& v)
		{
			version = v;
		}
	}
}