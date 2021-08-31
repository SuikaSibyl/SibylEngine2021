#include "SIByLpch.h"
#include "DX12VertexBuffer.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/Core/UploadBuffer.h"


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
		PROFILE_SCOPE_FUNCTION();

		m_VertexBuffer = CreateDefaultBuffer(sizeof(float) * number, vertices);
	}

	void DX12VertexBuffer::SetLayout(const VertexBufferLayout& layout)
	{
		PROFILE_SCOPE_FUNCTION();

		m_Layout = layout;
		m_VertexBufferView.BufferLocation = m_VertexBuffer->GetGPUVirtualAddress();
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