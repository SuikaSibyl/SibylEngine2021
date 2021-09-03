#pragma once

#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"

namespace SIByL
{
	class DX12TriangleMesh :public TriangleMesh
	{
	public:
		DX12TriangleMesh(
			float* vertices, uint32_t vCount,
			unsigned int* indices, uint32_t iCount,
			VertexBufferLayout layout);

		virtual void RasterDraw() override;

	protected:

	};
}