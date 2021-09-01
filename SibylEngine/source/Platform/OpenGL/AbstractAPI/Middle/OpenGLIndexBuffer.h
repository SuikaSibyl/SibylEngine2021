#pragma once

#include "Sibyl/Graphic/AbstractAPI/Middle/IndexBuffer.h"

namespace SIByL
{
	class OpenGLIndexBuffer :public IndexBuffer
	{
	public:
		OpenGLIndexBuffer(unsigned int* indices, uint32_t iCount);
		void SetData(unsigned int* indices, UINT32 number);
		virtual uint32_t Count() override { return m_Count; }

	private:
		unsigned int m_ElementBufferObject;
		uint32_t m_Count;
	};
}