#pragma once

#include <iostream>
#include "Sibyl/Graphic/Geometry/Vertex.h"
#include "Sibyl/Graphic/AbstractAPI/Middle/ShaderData.h"

namespace SIByL
{
	class VertexBufferLayout;

	////=============================================================================///

	class VertexBuffer
	{
	public:
		enum class Type
		{
			Static,
			Dynamic,
			Stream,
		};

		/////////////////////////////////////////////////////////
		///				    	Constructors		          ///
		static VertexBuffer* Create(float* vertices, uint32_t floatCount, Type type = Type::Static);
		virtual ~VertexBuffer() = default;

		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		virtual void SetLayout(const VertexBufferLayout& layout) = 0;
		virtual const VertexBufferLayout& GetLayout() = 0;
	};


	////=============================================================================///

	class VertexBufferLayout
	{
	public:
		/////////////////////////////////////////////////////////
		///				    	Constructors		          ///
		VertexBufferLayout() {}
		VertexBufferLayout(const std::initializer_list<BufferElement>&elements)
			:m_Elements(elements)
		{
			CalculateOffsetsAndStride();
		}

		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		inline const std::vector<BufferElement>& GetElements() const { return m_Elements; }
		inline const uint32_t GetStide() const { return m_Stride; }
		std::vector<BufferElement>::iterator begin() { return m_Elements.begin(); }
		std::vector<BufferElement>::iterator end() { return m_Elements.end(); }

	private:
		void CalculateOffsetsAndStride()
		{
			uint32_t offset = 0;
			m_Stride = 0;
			for (auto& element : m_Elements)
			{
				element.Offset = offset;
				offset += element.Size;
				m_Stride += element.Size;
			}
		}

	private:
		std::vector<BufferElement> m_Elements;
		uint32_t m_Stride = 0;
	};
}