#include "SIByLpch.h"
#include "ShaderRegister.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h"

namespace SIByL
{
	void ShaderRegister::RegisterAll()
	{
		RegisterUnlitTexture();
		RegisterLit();
		RegisterLitHair();

		RegisterEarlyZOpaque();
		RegisterEarlyZDither();

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
				{ShaderResourceType::Texture2D, "u_Main"},
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
				{ShaderResourceType::Texture2D, "u_Main"},
				{ShaderResourceType::Texture2D, "u_Normal"},
			},
		};

		Shader::Create("Shaders/SIByL/Lit",
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 2 }),
			ShaderBinderDesc(CBlayouts, SRlayouts));
	}
	

	void ShaderRegister::RegisterLitHair()
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
				{ShaderResourceType::Texture2D, "u_Main"},
				{ShaderResourceType::Texture2D, "u_Normal"},
				{ShaderResourceType::Texture2D, "u_DiffuseAO"},
				{ShaderResourceType::Texture2D, "u_IBLLUT"},
			},
			{
				{ShaderResourceType::Cubemap, "u_SpecularCube"},
			}
		};

		Shader::Create("Shaders/SIByL/LitHair",
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 2 }),
			ShaderBinderDesc(CBlayouts, SRlayouts));
	}

	void ShaderRegister::RegisterEarlyZOpaque()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			// ConstantBuffer0 : Per Object
			ConstantBufferLayout::PerObjectConstants,
			// ConstantBuffer1 : Per Material
			{

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
			},
		};

		Shader::Create("Shaders/SIByL/EarlyZOpaque",
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 1 }),
			ShaderBinderDesc(CBlayouts, SRlayouts));
	}

	void ShaderRegister::RegisterEarlyZDither()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			// ConstantBuffer0 : Per Object
			ConstantBufferLayout::PerObjectConstants,
			// ConstantBuffer1 : Per Material
			{

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
				{ShaderResourceType::Texture2D, "u_Main"},
			},
		};

		Shader::Create("Shaders/SIByL/EarlyZDither",
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 1 }),
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