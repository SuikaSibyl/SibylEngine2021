#include "SIByLpch.h"
#include "OpenGLVertexBuffer.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"

namespace SIByL
{
	OpenGLVertexBuffer::OpenGLVertexBuffer()
	{
		// Create VBO
		glGenBuffers(1, &m_VertexBufferObject);
		glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferObject);

	}

}