#pragma once

#include <iostream>

namespace SIByL
{
	class IndexBuffer
	{
	public:
		static IndexBuffer* Create(unsigned int* indices, uint32_t iCount);
		virtual ~IndexBuffer() {}
		virtual uint32_t Count() = 0;
	};
}