#pragma once

#include "Primitive.h"
#include "Sibyl/Graphic/AbstractAPI/Middle/VertexBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Middle/IndexBuffer.h"

namespace SIByL
{
	class TriangleMesh :public Primitive
	{
	public:
		static Ref<TriangleMesh> Create(
			float* vertices, uint32_t vCount,
			unsigned int* indices, uint32_t iCount,
			VertexBufferLayout layout);
		virtual void RasterDraw() = 0;

	protected:
		Ref<VertexBuffer>	m_VertexBuffer;
		Ref<IndexBuffer>	m_IndexBuffer;
	};
}