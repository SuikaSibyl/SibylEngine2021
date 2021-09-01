#pragma once

#include "Sibyl/Graphic/AbstractAPI/Middle/IndexBuffer.h"

#include "Platform/DirectX12/AbstractAPI/Bottom/DX12Resource.h"

namespace SIByL
{
	class DX12IndexBuffer :public IndexBuffer
	{
	public:
		DX12IndexBuffer(unsigned int* indices, uint32_t iCount);
		void SetData(unsigned int* indices, UINT32 number);
		virtual uint32_t Count() override { return m_Count; }
		const D3D12_INDEX_BUFFER_VIEW& GetIndexBufferView() { return m_IndexBufferView; }

	private:
		uint32_t m_Count;
		Ref<DX12Resource> m_Resource;
		D3D12_INDEX_BUFFER_VIEW m_IndexBufferView;
	};
}