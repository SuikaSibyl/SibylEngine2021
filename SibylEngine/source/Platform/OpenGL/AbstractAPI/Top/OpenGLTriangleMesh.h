#pragma once

#include "Sibyl/Graphic/Core/Geometry/TriangleMesh.h"

namespace SIByL
{
	class OpenGLTriangleMesh :public TriangleMesh
	{
	public:
		OpenGLTriangleMesh(
			float* vertices, uint32_t vCount,
			unsigned int* indices, uint32_t iCount,
			VertexBufferLayout layout);

		OpenGLTriangleMesh(
			const std::vector<MeshData>& meshDatas,
			VertexBufferLayout layout);

		virtual void RasterDraw() override;
		virtual void RasterDrawSubmeshStart() override;
		virtual void RasterDrawSubmesh(SubMesh& submesh) override;

	protected:
		unsigned int m_VertexArrayObject;

		struct OpenGLSubmesh
		{
			unsigned int		m_VertexArrayObject;
			Ref<VertexBuffer>	m_VertexBuffer;
			Ref<IndexBuffer>	m_IndexBuffer;
		};
		std::vector<OpenGLSubmesh> Submeshes;
	};
}