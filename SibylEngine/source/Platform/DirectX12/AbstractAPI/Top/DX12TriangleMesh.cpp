#include "SIByLpch.h"
#include "DX12TriangleMesh.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12VertexBuffer.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12IndexBuffer.h"
#include "Sibyl/Graphic/Core/Geometry/MeshLoader.h"

namespace SIByL
{
	DX12TriangleMesh::DX12TriangleMesh(
		float* vertices, uint32_t vCount, 
		unsigned int* indices, uint32_t iCount, 
		VertexBufferLayout layout)
	{
		PROFILE_SCOPE_FUNCTION();

		// Bind Vertex Buffer & IndexBuffer
		m_VertexBuffer.reset(VertexBuffer::Create(vertices, vCount));
		m_VertexBuffer->SetLayout(layout);
		m_IndexBuffer.reset(IndexBuffer::Create(indices, iCount));

		// Submesh
		m_SubMeshes.push_back(SubMesh({ 0,0,iCount }));
	}

	DX12TriangleMesh::DX12TriangleMesh(
		const std::vector<MeshData>& meshDatas,
		VertexBufferLayout layout)
	{
		std::vector<float> vertices;
		std::vector<uint32_t> indices;
		uint32_t vCount = 0, iCount = 0;

		//
		for (const MeshData& data : meshDatas)
		{
			m_SubMeshes.push_back(SubMesh({
				(uint32_t)((uint32_t)vertices.size() * sizeof(float) / layout.GetStide()),
				(uint32_t)((uint32_t)indices.size()),
				(uint32_t)data.indices.size() }));

			vertices.insert(vertices.end(),
				std::begin(data.vertices),
				std::end(data.vertices));
			vCount += data.vNum;

			indices.insert(indices.end(),
				std::begin(data.indices),
				std::end(data.indices));
			iCount += data.iNum;
		}

		// Bind Vertex Buffer & IndexBuffer
		m_VertexBuffer.reset(VertexBuffer::Create(vertices.data(), vertices.size()));
		m_VertexBuffer->SetLayout(layout);
		m_IndexBuffer.reset(IndexBuffer::Create(indices.data(), iCount));
	}

	void DX12TriangleMesh::RasterDraw()
	{
		PROFILE_SCOPE_FUNCTION();

		ID3D12GraphicsCommandList* cmdList = DX12Context::GetInFlightDXGraphicCommandList();
		DX12VertexBuffer* dxVertexBuffer = dynamic_cast<DX12VertexBuffer*>(m_VertexBuffer.get());
		cmdList->IASetVertexBuffers(0, 1, &(dxVertexBuffer->GetVertexBufferView()));
		DX12IndexBuffer* dxIndexBuffer = dynamic_cast<DX12IndexBuffer*>(m_IndexBuffer.get());
		cmdList->IASetIndexBuffer(&(dxIndexBuffer->GetIndexBufferView()));
		cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		cmdList->DrawIndexedInstanced(m_IndexBuffer->Count(), 1, 0, 0, 0);
	}

	void DX12TriangleMesh::RasterDrawSubmeshStart()
	{
		ID3D12GraphicsCommandList* cmdList = DX12Context::GetInFlightDXGraphicCommandList();
		DX12VertexBuffer* dxVertexBuffer = dynamic_cast<DX12VertexBuffer*>(m_VertexBuffer.get());
		cmdList->IASetVertexBuffers(0, 1, &(dxVertexBuffer->GetVertexBufferView()));
		DX12IndexBuffer* dxIndexBuffer = dynamic_cast<DX12IndexBuffer*>(m_IndexBuffer.get());
		cmdList->IASetIndexBuffer(&(dxIndexBuffer->GetIndexBufferView()));
		cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	}

	void DX12TriangleMesh::RasterDrawSubmesh(SubMesh& submesh)
	{
		ID3D12GraphicsCommandList* cmdList = DX12Context::GetInFlightDXGraphicCommandList();
		cmdList->DrawIndexedInstanced(submesh.IndexNumber, 1, submesh.IndexLocation, submesh.VertexLocation, 0);
	}
}