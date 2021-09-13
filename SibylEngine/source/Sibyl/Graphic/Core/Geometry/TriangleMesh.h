#pragma once

#include "Primitive.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/VertexBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/IndexBuffer.h"
#include "Sibyl/Graphic/Core/Geometry/MeshLoader.h"

namespace SIByL
{
	struct SubMesh
	{
		uint32_t VertexLocation;
		uint32_t IndexLocation;
		uint32_t IndexNumber;
	};

	class TriangleMesh :public Primitive
	{
	public:
		static Ref<TriangleMesh> Create(
			float* vertices, uint32_t vCount,
			unsigned int* indices, uint32_t iCount,
			VertexBufferLayout layout);		
		
		static Ref<TriangleMesh> Create(
			const std::vector<MeshData>& meshDatas,
			VertexBufferLayout layout, const std::string& path);

		using iter = std::vector<SubMesh>::iterator;
		iter begin() { return m_SubMeshes.begin(); }
		iter end() { return m_SubMeshes.end(); }
		UINT GetSubmesh();

		virtual void RasterDraw() = 0;
		virtual void RasterDrawSubmeshStart() = 0;
		virtual void RasterDrawSubmesh(SubMesh& submesh) = 0;

		std::string m_Path;

	protected:
		Ref<VertexBuffer>	m_VertexBuffer;
		Ref<IndexBuffer>	m_IndexBuffer;
		std::vector<SubMesh> m_SubMeshes;
	};
}