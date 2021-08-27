#pragma once

namespace SIByL
{
	enum class ShaderDataType
	{
		None,
		Float,
		Float2,
		Float3,
		Float4,
		Mat3,
		Mat4,
		Int,
		Int2,
		Int3,
		Int4,
		Bool,
	};

	enum class ShaderResourceType
	{
		Texture2D,
		Cubemap,
	};

	static uint32_t ShaderDataTypeSize(ShaderDataType type)
	{
		switch (type)
		{
		case SIByL::ShaderDataType::None:	return 0;
		case SIByL::ShaderDataType::Float:	return 4;
		case SIByL::ShaderDataType::Float2:	return 4 * 2;
		case SIByL::ShaderDataType::Float3:	return 4 * 3;
		case SIByL::ShaderDataType::Float4:	return 4 * 4;
		case SIByL::ShaderDataType::Mat3:	return 4 * 3 * 3;
		case SIByL::ShaderDataType::Mat4:	return 4 * 4 * 4;
		case SIByL::ShaderDataType::Int:	return 4;
		case SIByL::ShaderDataType::Int2:	return 4 * 2;
		case SIByL::ShaderDataType::Int3:	return 4 * 3;
		case SIByL::ShaderDataType::Int4:	return 4 * 4;
		case SIByL::ShaderDataType::Bool:	return 1;
		default:return 0;
		}
	}

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

	class ConstantBufferLayout
	{
	public:
		ConstantBufferLayout() {}
		ConstantBufferLayout(const std::initializer_list<BufferElement>& elements)
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

	struct TextureTableElement
	{
		ShaderResourceType Type;
		std::string Name;
	};

	class ShaderResourceLayout
	{
	public:
		ShaderResourceLayout() {}
		ShaderResourceLayout(const std::initializer_list<TextureTableElement>& elements)
			:m_Elements(elements)
		{
		}

		int SrvCount()
		{
			int count = 0;
			for (TextureTableElement& e : m_Elements)
			{
				count += 1;
			}
			return count;
		}

		inline const std::vector<TextureTableElement>& GetElements() const { return m_Elements; }

		std::vector<TextureTableElement>::iterator begin() { return m_Elements.begin(); }
		std::vector<TextureTableElement>::iterator end() { return m_Elements.end(); }

	private:
		std::vector<TextureTableElement> m_Elements;
	};
}