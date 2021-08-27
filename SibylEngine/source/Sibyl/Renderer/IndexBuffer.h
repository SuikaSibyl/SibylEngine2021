#pragma once

#include <iostream>

namespace SIByL
{
	class IndexBuffer
	{
	public:
		virtual ~IndexBuffer() {}
		static IndexBuffer* Create(unsigned int* indices, uint32_t iCount);
		virtual uint32_t Count() = 0;
	};
}