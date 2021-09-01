#pragma once

#include "Sibyl/Graphic/AbstractAPI/Middle/VertexBuffer.h"

namespace SIByL
{
	class OpenGLVertexBuffer : public VertexBuffer
	{
	public:
		/////////////////////////////////////////////////////////
		///				    	Constructors		          ///
		OpenGLVertexBuffer(float* vertices, uint32_t vCount, Type type = Type::Static);
		virtual ~OpenGLVertexBuffer() = default;

		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		virtual void SetLayout(const VertexBufferLayout& layout) override;
		virtual const VertexBufferLayout& GetLayout() override;

	protected:
		/////////////////////////////////////////////////////////
		///				     Local Function  		          ///
		void SetData(float* vertices, UINT32 number, Type type = Type::Static);

	private:
		/////////////////////////////////////////////////////////
		///				     Data Storage   		          ///
		uint32_t m_FloatCount;
		uint32_t m_Size;
		
		unsigned int m_VertexBufferObject;
		VertexBufferLayout m_Layout;
	};
}