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
		RegisterShadowOpaque();
		RegisterShadowDither();

		RegisterACES();

		RegisterTAA();
		RegisterFXAA();
		RegisterSharpen();
		RegisterVignette();
		RegisterSSAO();
		RegisterSSAOCombine();
		RegisterMedianBlur();

		RegisterBloomExtract();
		RegisterBloomCombine();
		RegisterBlur();
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
				{ShaderResourceType::Texture2D, "u_DirectionalShadowmap"},
			},
		};

		Shader::Create("Shaders/SIByL/Lit",
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 3 }),
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
				{ShaderResourceType::Texture2D, "u_DirectionalShadowmap"},
			},
			{
				{ShaderResourceType::Cubemap, "u_SpecularCube"},
			}
		};

		Shader::Create("Shaders/SIByL/LitHair",
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 3 }),
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
	
	void ShaderRegister::RegisterShadowOpaque()
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

		Shader::Create("Shaders/SIByL/ShadowOpaque",
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 1 }),
			ShaderBinderDesc(CBlayouts, SRlayouts));
	}

	void ShaderRegister::RegisterShadowDither()
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

		Shader::Create("Shaders/SIByL/ShadowDither",
			ShaderDesc({ true,SIByL::VertexBufferLayout::StandardVertexBufferLayout, 1 }),
			ShaderBinderDesc(CBlayouts, SRlayouts));

	}

	void ShaderRegister::RegisterACES()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float2, "OutputSize"},
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
				{ShaderDataType::Float2, "OutputSize"},
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
	
	void ShaderRegister::RegisterFXAA()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float2, "OutputSize"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_Texture"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "FXAAResult"},
			},
		};


		ComputeShader::Create("Shaders/Compute/FXAA",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}
	
	void ShaderRegister::RegisterSharpen()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float2, "OutputSize"},
				{ShaderDataType::Float, "uSharpFactor"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_Texture"},
				{ShaderResourceType::Texture2D, "u_Depth"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "SharpenResult"},
			},
		};


		ComputeShader::Create("Shaders/Compute/Sharpen",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}

	void ShaderRegister::RegisterMedianBlur()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float2, "OutputSize"},
				{ShaderDataType::Float, "u_radius"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_Input"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "MedianBlurResult"},
			},
		};

		ComputeShader::Create("Shaders/Compute/MedianBlurV",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
		ComputeShader::Create("Shaders/Compute/MedianBlurH",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}
	
	void ShaderRegister::RegisterVignette()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float2, "OutputSize"},
				{ShaderDataType::Float2, "uLensRadius"},
				{ShaderDataType::Float, "uFrameMod"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_Texture"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "VignetteResult"},
			},
		};


		ComputeShader::Create("Shaders/Compute/Vignette",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}
	
	void ShaderRegister::RegisterSSAO()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Mat4, "uProjection"},
				{ShaderDataType::Float2, "OutputSize"},
				{ShaderDataType::Float, "uRadius"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_Normal"},
				{ShaderResourceType::Texture2D, "u_Depth"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "SSAOResult"},
			},
		};


		ComputeShader::Create("Shaders/Compute/SSAO",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}
	
	void ShaderRegister::RegisterSSAOCombine()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float2, "OutputSize"},
				{ShaderDataType::Float, "uAOFactor"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_Texture"},
				{ShaderResourceType::Texture2D, "u_SSAO"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "SSAOCombinedResult"},
			},
		};


		ComputeShader::Create("Shaders/Compute/SSAOCombine",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}

	void ShaderRegister::RegisterBloomExtract()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float2, "OutputSize"},
				{ShaderDataType::Float, "uBloomThreshold"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_Texture"},
				{ShaderResourceType::Texture2D, "u_Depth"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "ExtractResult"},
			},
		};


		ComputeShader::Create("Shaders/Compute/BloomExtract",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}

	void ShaderRegister::RegisterBloomCombine()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float2, "OutputSize"},
				{ShaderDataType::Float, "uBloomFactor"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_Texture"},
				{ShaderResourceType::Texture2D, "TextureBloomBlur1"},
				{ShaderResourceType::Texture2D, "TextureBloomBlur2"},
				{ShaderResourceType::Texture2D, "TextureBloomBlur3"},
				{ShaderResourceType::Texture2D, "TextureBloomBlur4"},
				{ShaderResourceType::Texture2D, "TextureBloomBlur5"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "CombineResult"},
			},
		};


		ComputeShader::Create("Shaders/Compute/BloomCombine",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}

	void ShaderRegister::RegisterBlur()
	{
		std::vector<ConstantBufferLayout> CBlayouts =
		{
			{
				{ShaderDataType::Float2, "OutputSize"},
				{ShaderDataType::Float2, "uGlobalTexSize"},
				{ShaderDataType::Float2, "uTextureBlurInputSize"},
				{ShaderDataType::Float2, "uBlurDir"},
			}
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "u_Input"},
			},
		};

		std::vector<ComputeOutputLayout> COlayouts =
		{
			// Compute Output
			{
				{ShaderResourceType::Texture2D, "BlurResult"},
			},
		};

		ComputeShader::Create("Shaders/Compute/BlurLevel0",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
		ComputeShader::Create("Shaders/Compute/BlurLevel1",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
		ComputeShader::Create("Shaders/Compute/BlurLevel2",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
		ComputeShader::Create("Shaders/Compute/BlurLevel3",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
		ComputeShader::Create("Shaders/Compute/BlurLevel4",
			ShaderBinderDesc(CBlayouts, SRlayouts, COlayouts));
	}
}