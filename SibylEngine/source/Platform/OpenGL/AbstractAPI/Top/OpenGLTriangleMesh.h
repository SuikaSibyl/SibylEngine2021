#pragma once

#include "Sibyl/Graphic/Geometry/TriangleMesh.h"

namespace SIByL
{
	class OpenGLTriangleMesh :public TriangleMesh
	{
	public:
		OpenGLTriangleMesh(
			float* vertices, uint32_t vCount,
			unsigned int* indices, uint32_t iCount,
			VertexBufferLayout layout);

		virtual void RasterDraw() override;

	protected:
		unsigned int m_VertexArrayObject;
	};
}