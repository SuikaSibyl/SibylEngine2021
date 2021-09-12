#pragma once

#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"

namespace SIByL
{
	class MeshData;

	class DX12TriangleMesh :public TriangleMesh
	{
	public:
		DX12TriangleMesh(
			float* vertices, uint32_t vCount,
			unsigned int* indices, uint32_t iCount,
			VertexBufferLayout layout);

		DX12TriangleMesh(
			const std::vector<MeshData>& meshDatas,
			VertexBufferLayout layout);

		virtual void RasterDraw() override;
		virtual void RasterDrawSubmeshStart() override;
		virtual void RasterDrawSubmesh(SubMesh& submesh) override;

	protected:

	};
}