#pragma once

#include "Primitive.h"
#include "Sibyl/Renderer/VertexBuffer.h"
#include "Sibyl/Renderer/IndexBuffer.h"

namespace SIByL
{
	class TriangleMesh :public Primitive
	{
	public:
		static TriangleMesh* Create(
			float* vertices, uint32_t vCount,
			unsigned int* indices, uint32_t iCount,
			VertexBufferLayout layout);
		virtual void RasterDraw() = 0;

	protected:
		std::unique_ptr<VertexBuffer>	m_VertexBuffer;
		std::unique_ptr<IndexBuffer>	m_IndexBuffer;
	};
}