#include "SIByLpch.h"
#include "DX12VertexBuffer.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12UploadBuffer.h"


namespace SIByL
{
	DX12VertexBuffer::DX12VertexBuffer(float* vertices, uint32_t floatCount, Type type)
	{
		PROFILE_SCOPE_FUNCTION();

		m_FloatCount = floatCount;
		SetData(vertices, floatCount, type);
	}

	void DX12VertexBuffer::SetData(float* vertices, UINT32 number, Type type)
	{
		m_Resource.reset(new DX12Resource(CreateDefaultBuffer(sizeof(float) * number, vertices), L"VertexBuffer"));
	}

	void DX12VertexBuffer::SetLayout(const VertexBufferLayout& layout)
	{
		PROFILE_SCOPE_FUNCTION();

		m_Layout = layout;
		m_VertexBufferView.BufferLocation = m_Resource->GetD3D12Resource()->GetGPUVirtualAddress();
		m_VertexBufferView.SizeInBytes = sizeof(float) * m_FloatCount;
		m_VertexBufferView.StrideInBytes = layout.GetStide();
	}

	const VertexBufferLayout& DX12VertexBuffer::GetLayout()
	{
		return m_Layout;
	}

	const D3D12_VERTEX_BUFFER_VIEW& DX12VertexBuffer::GetVertexBufferView()
	{
		return m_VertexBufferView;
	}
}