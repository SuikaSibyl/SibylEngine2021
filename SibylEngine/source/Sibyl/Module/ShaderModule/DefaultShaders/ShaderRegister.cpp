#include "SIByLpch.h"
#include "ShaderRegister.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h"

namespace SIByL
{
	void ShaderRegister::RegisterAll()
	{
		RegisterUnlitTexture();

		RegisterACES();
	}

	void ShaderRegister::RegisterUnlitTexture()
	{
		VertexBufferLayout layout =
		{
			{ShaderDataType::Float3, "POSITION"},
			{ShaderDataType::Float2, "TEXCOORD"},
		};

		std::vector<ConstantBufferLayout> CBlayouts =
		{
			// ConstantBuffer0 : Per Object
			ConstantBufferLayout::PerObjectConstants,
			// ConstantBuffer1 : Per Material
			{
				{ShaderDataType::RGBA, "Color"},
			},
			// ConstantBuffer2 : Per Camera
			ConstantBufferLayout::PerCameraConstants,
			// ConstantBuffer3 : Per Frame
			ConstantBufferLayout::PerFrameConstants,
		};


		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "Main"},
			},
		};

		Shader::Create("Shaders/SIByL/Texture",
			ShaderDesc({ true,layout, 2 }),
			ShaderBinderDesc(CBlayouts, SRlayouts));
	}


	void ShaderRegister::RegisterACES()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::RGBA, "Color"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "Main"},
			},
		};

		ComputeShader::Create("Shaders/Compute/ACES",
			ShaderBinderDesc(CBlayouts, SRlayouts));
	}
}