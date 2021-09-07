#include "SIByLpch.h"
#include "ShaderData.h"

namespace SIByL
{
	ConstantBufferLayout ConstantBufferLayout::PerObjectConstants =
	{
		{ShaderDataType::Mat4, "Model"},
	};

	ConstantBufferLayout ConstantBufferLayout::PerCameraConstants =
	{
		{ShaderDataType::Mat4, "View"},
		{ShaderDataType::Mat4, "Projection" },

	};

	ConstantBufferLayout ConstantBufferLayout::PerFrameConstants =
	{
		{ShaderDataType::Float4, "LightColor"},
	};
}