#include "SIByLpch.h"
#include "OpenGLTriangleMesh.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"

namespace SIByL
{
	OpenGLTriangleMesh::OpenGLTriangleMesh(
		float* vertices, uint32_t floatCount,
		unsigned int* indices, uint32_t iCount,
		VertexBufferLayout layout)
	{
		PROFILE_SCOPE_FUNCTION();

		// Create VAO
		glGenVertexArrays(1, &m_VertexArrayObject);
		glBindVertexArray(m_VertexArrayObject);

		// Bind Vertex Buffer & IndexBuffer
		m_VertexBuffer.reset(VertexBuffer::Create(vertices, floatCount));
		m_VertexBuffer->SetLayout(layout);
		m_IndexBuffer.reset(IndexBuffer::Create(indices, iCount));
	}

	void OpenGLTriangleMesh::RasterDraw()
	{
		PROFILE_SCOPE_FUNCTION();

		glBindVertexArray(m_VertexArrayObject);
		glDrawElements(GL_TRIANGLES, m_IndexBuffer->Count() , GL_UNSIGNED_INT, 0);
	}
}