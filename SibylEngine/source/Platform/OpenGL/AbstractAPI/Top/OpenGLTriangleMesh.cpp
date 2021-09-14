#include "SIByLpch.h"
#include "OpenGLTriangleMesh.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"

namespace SIByL
{
	OpenGLTriangleMesh::OpenGLTriangleMesh(
		float* vertices, uint32_t floatCount,
		unsigned int* indices, uint32_t iCount,
		VertexBufferLayout layout)
	{
		int index = 0;

		OpenGLSubmesh submesh;

		// Create VAO
		glGenVertexArrays(1, &submesh.m_VertexArrayObject);
		glBindVertexArray(submesh.m_VertexArrayObject);

		// Bind Vertex Buffer & IndexBuffer
		submesh.m_VertexBuffer.reset(VertexBuffer::Create(vertices, floatCount));
		submesh.m_VertexBuffer->SetLayout(layout);
		submesh.m_IndexBuffer.reset(IndexBuffer::Create(indices, iCount));

		Submeshes.emplace_back(submesh);
		SubMesh tmp; tmp.Index = index++;
		m_SubMeshes.emplace_back(tmp);
	}

	OpenGLTriangleMesh::OpenGLTriangleMesh(
		const std::vector<MeshData>& meshDatas,
		VertexBufferLayout layout)
	{
		int index = 0;

		for (MeshData meshData : meshDatas)
		{
			OpenGLSubmesh submesh;

			// Create VAO
			glGenVertexArrays(1, &submesh.m_VertexArrayObject);
			glBindVertexArray(submesh.m_VertexArrayObject);

			// Bind Vertex Buffer & IndexBuffer
			submesh.m_VertexBuffer.reset(VertexBuffer::Create(meshData.vertices.data(), meshData.vertices.size()));
			submesh.m_VertexBuffer->SetLayout(layout);
			submesh.m_IndexBuffer.reset(IndexBuffer::Create(meshData.indices.data(), meshData.indices.size()));

			Submeshes.emplace_back(submesh);
			SubMesh tmp; tmp.Index = index++;
			m_SubMeshes.emplace_back(tmp);
		}
	}

	void OpenGLTriangleMesh::RasterDraw()
	{
		glBindVertexArray(Submeshes[0].m_VertexArrayObject);
		glDrawElements(GL_TRIANGLES, Submeshes[0].m_IndexBuffer->Count(), GL_UNSIGNED_INT, 0);
	}

	void OpenGLTriangleMesh::RasterDrawSubmeshStart()
	{

	}

	void OpenGLTriangleMesh::RasterDrawSubmesh(SubMesh& submesh)
	{
		glBindVertexArray(Submeshes[submesh.Index].m_VertexArrayObject);
		glDrawElements(GL_TRIANGLES, Submeshes[submesh.Index].m_IndexBuffer->Count(), GL_UNSIGNED_INT, 0);
	}

}