#pragma once

#include <filesystem>
#include "SStructuredBuffer.h"
#include "SFileSystem.h"

namespace SIByL
{
	namespace File
	{
		class SCache
		{
		public:
			SCache(const std::filesystem::path& path);

			void Serialize();
			void Deserialize();

			TimeStamp GetVersion();
			void SetVersion(const TimeStamp& v);

			template <class T>
			void PushStruct(const T& data);

			template <class T>
			void PullStruct(T& data);

			template <class T>
			void PushStructVector(const std::vector<T>& data);

			template <class T>
			void PullStructVector(std::vector<T>& data);
			
		private:
			TimeStamp version;
			unsigned int bufferCount = 0;
			std::filesystem::path path;
			std::vector<SStructuredBuffer> mBuffers;
		};


		template <class T>
		void SCache::PushStruct(const T& data)
		{
			bufferCount++;
			SStructuredBuffer sbuffer;
			sbuffer.PushStruct<T>(data);
			mBuffers.emplace_back(sbuffer);
		}

		template <class T>
		void SCache::PullStruct(T& data)
		{
			mBuffers[bufferCount].PullStruct<T>(data);
			mBuffers[bufferCount].ReleaseBuffer();
			bufferCount++;
		}

		template <class T>
		void SCache::PushStructVector(const std::vector<T>& data)
		{
			bufferCount++;
			SStructuredBuffer sbuffer;
			sbuffer.PushStructVector<T>(data);
			mBuffers.emplace_back(sbuffer);
		}

		template <class T>
		void SCache::PullStructVector(std::vector<T>& data)
		{
			mBuffers[bufferCount].PullStructVector<T>(data);
			mBuffers[bufferCount].ReleaseBuffer();
			bufferCount++;
		}
	}
}