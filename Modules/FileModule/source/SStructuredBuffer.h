#pragma once

#include <Core/module.h>
#include <vector>

namespace SIByL
{
	namespace File
	{
		class SStructuredBuffer
		{
		public:
			~SStructuredBuffer();
			void ReleaseBuffer();

			void Serialize(std::ofstream& ofstream);
			void Deserialize(std::ifstream& ifstream);

			template <class T>
			void PushStruct(const T& data);

			template <class T>
			void PullStruct(T& data);

			template <class T>
			void PushStructVector(const std::vector<T>& data);

			template <class T>
			void PullStructVector(std::vector<T>& data);

		private:
			unsigned int structureSize;
			unsigned int structureCount;
			unsigned int byteSize;

			char* buffer = nullptr;
		};

		template <class T>
		void SStructuredBuffer::PushStruct(const T& data)
		{
			structureSize = sizeof(T);
			structureCount = 1;
			byteSize = structureSize;

			S_CORE_ASSERT(buffer == nullptr, "SStructedBuffer::PushStruct failed: buffer already filled!");
			buffer = new char[byteSize];
			memcpy(buffer, &data, byteSize);
		}

		template <class T>
		void SStructuredBuffer::PullStruct(T& data)
		{
			S_CORE_ASSERT(buffer != nullptr, "SStructedBuffer::PullStruct failed: buffer is not filled!");
			memcpy(&data, buffer, byteSize);
		}

		template <class T>
		void SStructuredBuffer::PushStructVector(const std::vector<T>& data)
		{
			structureSize = sizeof(T);
			structureCount = data.size();
			byteSize = data.size() * sizeof(T);

			S_CORE_ASSERT(buffer == nullptr, "SStructedBuffer::PushStruct failed: buffer already filled!");
			buffer = new char[byteSize];
			memcpy(buffer, data.data(), byteSize);
		}

		template <class T>
		void SStructuredBuffer::PullStructVector(std::vector<T>& data)
		{
			S_CORE_ASSERT(buffer != nullptr, "SStructedBuffer::PullStruct failed: buffer is not filled!");
			data.resize(structureCount);
			memcpy(data.data(), buffer, byteSize);
		}
	}
}