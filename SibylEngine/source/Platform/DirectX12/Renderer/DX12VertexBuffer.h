#pragma once

#include "Sibyl/Renderer/VertexBuffer.h"

namespace SIByL
{
	class VertexData;
	class DX12VertexBuffer : public VertexBuffer
	{
	public:
		DX12VertexBuffer(float* vertices, uint32_t vCount, Type type = Type::Static);
		void SetData(float* vertices, UINT32 number, Type type = Type::Static);
		virtual void SetLayout(const VertexBufferLayout& layout) override;
		virtual const VertexBufferLayout& GetLayout() override;

		const D3D12_VERTEX_BUFFER_VIEW& GetVertexBufferView();

	private:
		D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView;
		ComPtr<ID3D12Resource> m_VertexBuffer;
		VertexBufferLayout m_Layout;
	};
}