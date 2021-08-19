#pragma once

#include "Sibyl/Renderer/VertexBuffer.h"

namespace SIByL
{
	class VertexData;
	class OpenGLVertexBuffer :public VertexBuffer
	{
	public:
		OpenGLVertexBuffer(float* vertices, uint32_t vCount, Type type = Type::Static);
		void SetData(float* vertices, UINT32 number, Type type = Type::Static);
		virtual void SetLayout(const VertexBufferLayout& layout) override;
		virtual const VertexBufferLayout& GetLayout() override;

	private:
		unsigned int m_VertexBufferObject;
		VertexBufferLayout m_Layout;
	};
}