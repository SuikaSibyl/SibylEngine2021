#pragma once

#include "Sibyl/Renderer/VertexBuffer.h"

namespace SIByL
{
	class OpenGLVertexBuffer :public VertexBuffer
	{
	public:
		OpenGLVertexBuffer();

	private:
		unsigned int m_VertexBufferObject;
	};
}