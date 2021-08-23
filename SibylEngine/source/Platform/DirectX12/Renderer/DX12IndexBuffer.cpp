#include "SIByLpch.h"
#include "DX12IndexBuffer.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/Core/UploadBuffer.h"

namespace SIByL
{
	DX12IndexBuffer::DX12IndexBuffer(unsigned int* indices, uint32_t iCount)
	{
		m_Count = iCount;
		SetData(indices, iCount);
	}

	void DX12IndexBuffer::SetData(unsigned int* indices, UINT32 number)
	{
		m_IndexBuffer = CreateDefaultBuffer(sizeof(unsigned int) * number, indices);
		m_IndexBufferView.BufferLocation = m_IndexBuffer->GetGPUVirtualAddress();
		m_IndexBufferView.Format = DXGI_FORMAT_R32_UINT;
		m_IndexBufferView.SizeInBytes = sizeof(unsigned int) * number;
	}


}