#include "SIByLpch.h"
#include "OpenGLVertexBuffer.h"

#include "Sibyl/Graphic/Geometry/Vertex.h"
#include "Platform/OpenGL/Common/OpenGLContext.h"

namespace SIByL
{
	static GLenum ShaderDataTypeToOpenGLBaseType(ShaderDataType type)
	{
		switch (type)
		{
		case SIByL::ShaderDataType::None:	return GL_FLOAT;
		case SIByL::ShaderDataType::Float:	return GL_FLOAT;
		case SIByL::ShaderDataType::Float2:	return GL_FLOAT;
		case SIByL::ShaderDataType::Float3:	return GL_FLOAT;
		case SIByL::ShaderDataType::Float4:	return GL_FLOAT;
		case SIByL::ShaderDataType::Mat3:	return GL_FLOAT;
		case SIByL::ShaderDataType::Mat4:	return GL_FLOAT;
		case SIByL::ShaderDataType::Int:	return GL_INT;
		case SIByL::ShaderDataType::Int2:	return GL_INT;
		case SIByL::ShaderDataType::Int3:	return GL_INT;
		case SIByL::ShaderDataType::Int4:	return GL_INT;
		case SIByL::ShaderDataType::Bool:	return GL_BOOL;
		default:return 0;
		}
	}

	OpenGLVertexBuffer::OpenGLVertexBuffer(float* vertices, uint32_t vCount, Type type)
	{
		PROFILE_SCOPE_FUNCTION();

		// Create VBO
		glGenBuffers(1, &m_VertexBufferObject);
		SetData(vertices, vCount, type);
	}

	void OpenGLVertexBuffer::SetData(float* vertices, UINT32 number, Type type)
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_VertexBufferObject);
		int dataType = (type == Type::Static) ? (GL_STATIC_DRAW) : (
			(type == Type::Dynamic) ? (GL_DYNAMIC_DRAW) : (GL_STREAM_DRAW));
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * number, vertices, dataType);

	}

	void OpenGLVertexBuffer::SetLayout(const VertexBufferLayout& layout)
	{
		m_Layout = layout;
		uint32_t index = 0;
		for (const auto& element : m_Layout)
		{
			glVertexAttribPointer(index,
				element.GetComponentCount(),
				ShaderDataTypeToOpenGLBaseType(element.Type),
				element.Normalized ? GL_TRUE : GL_FALSE,
				layout.GetStide(),
				(void*)element.Offset);
			glEnableVertexAttribArray(index);
			index++;
		}
	}

	const VertexBufferLayout& OpenGLVertexBuffer::GetLayout()
	{
		return m_Layout;
	}
}