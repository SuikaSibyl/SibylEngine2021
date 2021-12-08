#include "SBuffer.h"

namespace SIByL
{
	namespace File
	{
		void SBuffer::MemcpyBack(char* input, unsigned int size)
		{
			unsigned int back = mBuffer.size();
			mBuffer.resize(mBuffer.size() + size);
			mCursor = &mBuffer[back];
			memcpy(mCursor, input, size);
		}
	}
}