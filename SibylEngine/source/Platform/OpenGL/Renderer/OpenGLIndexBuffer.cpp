#include "SIByLpch.h"
#include "OpenGLIndexBuffer.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"

namespace SIByL
{
	OpenGLIndexBuffer::OpenGLIndexBuffer(unsigned int* indices, uint32_t iCount)
	{
		PROFILE_SCOPE_FUNCTION();

		m_Count = iCount;
		glGenBuffers(1, &m_ElementBufferObject);
		SetData(indices, iCount);
	}

	void OpenGLIndexBuffer::SetData(unsigned int* indices, UINT32 number)
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ElementBufferObject);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, number * sizeof(unsigned int), indices, GL_STATIC_DRAW);
	}


}