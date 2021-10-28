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
		{ShaderDataType::Mat4, "PreviousPV" },
		{ShaderDataType::Mat4, "CurrentPV" },
		{ShaderDataType::Float4, "ViewPos" },
	};

	ConstantBufferLayout ConstantBufferLayout::PerFrameConstants =
	{
		{ShaderDataType::Int, "DirectionalLightNum"},
		{ShaderDataType::Int, "PointLightNum"},

		{ShaderDataType::Float3, "directionalLights[0].direction"},
		{ShaderDataType::Float,  "directionalLights[0].intensity"},
		{ShaderDataType::Float3, "directionalLights[0].color"},

		{ShaderDataType::Float3, "directionalLights[1].direction"},
		{ShaderDataType::Float,  "directionalLights[1].intensity"},
		{ShaderDataType::Float3, "directionalLights[1].color"},
		
		{ShaderDataType::Float3, "directionalLights[2].direction"},
		{ShaderDataType::Float,  "directionalLights[2].intensity"},
		{ShaderDataType::Float3, "directionalLights[2].color"},

		{ShaderDataType::Float3, "directionalLights[3].direction"},
		{ShaderDataType::Float,  "directionalLights[3].intensity"},
		{ShaderDataType::Float3, "directionalLights[3].color"},
	};
}