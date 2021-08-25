#pragma once

#include "Primitive.h"
#include "Sibyl/Renderer/ShaderData.h"

namespace SIByL
{
	struct BufferElement
	{
		std::string Name;
		ShaderDataType Type;
		uint32_t Offset;
		uint32_t Size;
		bool Normalized;

		BufferElement(ShaderDataType type, const std::string& name, bool normalized = false)
			:Name(name), Type(type), Size(ShaderDataTypeSize(type)), Offset(0), Normalized(normalized)
		{

		}

		uint32_t GetComponentCount() const
		{
			switch (Type)
			{
			case SIByL::ShaderDataType::None:	return 0;
			case SIByL::ShaderDataType::Float:	return 1;
			case SIByL::ShaderDataType::Float2:	return 2;
			case SIByL::ShaderDataType::Float3:	return 3;
			case SIByL::ShaderDataType::Float4:	return 4;
			case SIByL::ShaderDataType::Mat3:	return 3 * 3;
			case SIByL::ShaderDataType::Mat4:	return 4 * 4;
			case SIByL::ShaderDataType::Int:	return 1;
			case SIByL::ShaderDataType::Int2:	return 2;
			case SIByL::ShaderDataType::Int3:	return 3;
			case SIByL::ShaderDataType::Int4:	return 4;
			case SIByL::ShaderDataType::Bool:	return 1;
			default:return 0;
			}
		}
	};

	class VertexBufferLayout
	{
	public:
		VertexBufferLayout() {}
		VertexBufferLayout(const std::initializer_list<BufferElement>& elements)
			:m_Elements(elements)
		{
			CalculateOffsetsAndStride();
		}

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

	class Vertex :public Primitive
	{
	public:


	private:


	};
}