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
}