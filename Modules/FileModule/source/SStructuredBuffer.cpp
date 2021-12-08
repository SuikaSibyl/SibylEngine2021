#include "SStructuredBuffer.h"

#include <fstream>

namespace SIByL
{
	namespace File
	{
		SStructuredBuffer::~SStructuredBuffer()
		{

		}

		void SStructuredBuffer::ReleaseBuffer()
		{
			if (buffer != nullptr)
				delete[] buffer;
		}

		void SStructuredBuffer::Serialize(std::ofstream& ofstream)
		{
			char* bufferHead = new char[3 * sizeof(unsigned int)];
			unsigned int head[3] = { structureSize ,structureCount, byteSize };
			memcpy(bufferHead, head, 3 * sizeof(unsigned int));
			ofstream.write(bufferHead, 3 * sizeof(unsigned int));
			ofstream.write(buffer, byteSize);
			delete[] bufferHead;
		}

		void SStructuredBuffer::Deserialize(std::ifstream& ifstream)
		{
			char* bufferHead = new char[3 * sizeof(unsigned int)];
			unsigned int head[3] = { 0 ,0, 0 };
			ifstream.read(bufferHead, 3 * sizeof(unsigned int));
			memcpy(head, bufferHead, 3 * sizeof(unsigned int));
			structureSize = head[0];
			structureCount = head[1];
			byteSize = head[2];
			buffer = new char[byteSize];
			ifstream.read(buffer, byteSize);
		}

	}
}