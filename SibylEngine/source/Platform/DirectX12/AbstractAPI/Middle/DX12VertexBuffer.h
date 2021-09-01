#pragma once

#include "Sibyl/Graphic/AbstractAPI/Middle/VertexBuffer.h"

#include "Platform/DirectX12/AbstractAPI/Bottom/DX12Resource.h"

namespace SIByL
{
	class DX12VertexBuffer : public VertexBuffer
	{
	public:
		/////////////////////////////////////////////////////////
		///				    	Constructors		          ///
		DX12VertexBuffer(float* vertices, uint32_t vCount, Type type = Type::Static);

		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		virtual void SetLayout(const VertexBufferLayout& layout) override;
		virtual const VertexBufferLayout& GetLayout() override;

		/////////////////////////////////////////////////////////
		///				      DX12 Function	 	              ///
		const D3D12_VERTEX_BUFFER_VIEW& GetVertexBufferView();

	protected:
		/////////////////////////////////////////////////////////
		///				     Local Function  		          ///
		void SetData(float* vertices, UINT32 number, Type type = Type::Static);

	protected:
		/////////////////////////////////////////////////////////
		///				     Data Storage   		          ///
		uint32_t m_FloatCount;
		uint32_t m_Size;

		D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView;
		Ref<DX12Resource> m_Resource;
		VertexBufferLayout m_Layout;
	};
}