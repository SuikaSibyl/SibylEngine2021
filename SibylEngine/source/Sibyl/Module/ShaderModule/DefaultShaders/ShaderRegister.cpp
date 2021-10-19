#include "SIByLpch.h"
#include "ShaderRegister.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h"

namespace SIByL
{
	void ShaderRegister::RegisterAll()
	{
		RegisterUnlitTexture();
		RegisterLit();

		RegisterACES();

		RegisterTAA();
	}

	void ShaderRegister::RegisterUnlitTexture()
	{
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
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 2 }),
			ShaderBinderDesc(CBlayouts, SRlayouts));
	}

	void ShaderRegister::RegisterLit()
	{
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

		Shader::Create("Shaders/SIByL/Lit",
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 2 }),
			ShaderBinderDesc(CBlayouts, SRlayouts));
	}


	void ShaderRegister::RegisterACES()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float, "Para"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "Input"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "ACESResult"},
			},
		};

		ComputeShader::Create("Shaders/Compute/ACES",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}

	void ShaderRegister::RegisterTAA()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float, "Alpha"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_PreviousFrame"},
				{ShaderResourceType::Texture2D, "u_CurrentFrame"},
				{ShaderResourceType::Texture2D, "u_Offset"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "TAAResult"},
			},
		};


		ComputeShader::Create("Shaders/Compute/TAA",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}
}