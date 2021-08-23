#include "SIByLpch.h"
#include "DX12TriangleMesh.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/Renderer/DX12VertexBuffer.h"
#include "Platform/DirectX12/Renderer/DX12IndexBuffer.h"

namespace SIByL
{
	DX12TriangleMesh::DX12TriangleMesh(
		float* vertices, uint32_t vCount, 
		unsigned int* indices, uint32_t iCount, 
		VertexBufferLayout layout)
	{
		// Bind Vertex Buffer & IndexBuffer
		m_VertexBuffer.reset(VertexBuffer::Create(vertices, vCount));
		m_VertexBuffer->SetLayout(layout);
		m_IndexBuffer.reset(IndexBuffer::Create(indices, iCount));
	}

	void DX12TriangleMesh::RasterDraw()
	{
		ID3D12GraphicsCommandList* cmdList = DX12Context::GetDXGraphicCommandList();
		DX12VertexBuffer* dxVertexBuffer = dynamic_cast<DX12VertexBuffer*>(m_VertexBuffer.get());
		cmdList->IASetVertexBuffers(0, 1, &(dxVertexBuffer->GetVertexBufferView()));
		DX12IndexBuffer* dxIndexBuffer = dynamic_cast<DX12IndexBuffer*>(m_IndexBuffer.get());
		cmdList->IASetIndexBuffer(&(dxIndexBuffer->GetIndexBufferView()));
		cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		cmdList->DrawIndexedInstanced(m_IndexBuffer->Count(), 1, 0, 0, 0);
	}
}