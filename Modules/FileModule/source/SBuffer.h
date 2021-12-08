#pragma once

#include <vector>

namespace SIByL
{
	namespace File
	{
		class SBuffer
		{
		public:
			void MemcpyBack(char* input, unsigned int size);

		private:
			char* mCursor;
			std::vector<char> mBuffer;
		};
	}
}