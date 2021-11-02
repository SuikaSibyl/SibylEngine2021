#pragma once

#include "SIByLpch.h"
#include "Bloom.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/ComputeInstance.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"


namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipeBloom::Build()
		{
			// Create Compute Shader
			BloomExtractInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BloomExtract"));
			BloomCombineInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BloomCombine"));
			BlurLevel0InstanceV = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel0"));
			BlurLevel0InstanceH = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel0"));
			BlurLevel1InstanceV = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel1"));
			BlurLevel1InstanceH = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel1"));
			BlurLevel2InstanceV = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel2"));
			BlurLevel2InstanceH = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel2"));
			BlurLevel3InstanceV = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel3"));
			BlurLevel3InstanceH = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel3"));
			BlurLevel4InstanceV = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel4"));
			BlurLevel4InstanceH = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\BlurLevel4"));

			// FrameBufferDesc desc;
			FrameBufferDesc desc;
			desc.Width = 1280;
			desc.Height = 720;
			desc.Formats = { FrameBufferTextureFormat::RGB8 };
			desc.ScaleX = 1;
			desc.ScaleY = 1;

			m_FrameBuffer_Bloom[0] = FrameBuffer::Create(desc, "BloomExtract");
			m_FrameBuffer_Bloom[11] = FrameBuffer::Create(desc, "BloomCombine");

			desc.ScaleX = 0.5;
			desc.ScaleY = 0.5;
			m_FrameBuffer_Bloom[1] = FrameBuffer::Create(desc, "BloomBlur0v");
			m_FrameBuffer_Bloom[2] = FrameBuffer::Create(desc, "BloomBlur0h");

			desc.ScaleX = 0.25;
			desc.ScaleY = 0.25;
			m_FrameBuffer_Bloom[3] = FrameBuffer::Create(desc, "BloomBlur1v");
			m_FrameBuffer_Bloom[4] = FrameBuffer::Create(desc, "BloomBlur1h");

			desc.ScaleX = 0.125;
			desc.ScaleY = 0.125;
			m_FrameBuffer_Bloom[5] = FrameBuffer::Create(desc, "BloomBlur2v");
			m_FrameBuffer_Bloom[6] = FrameBuffer::Create(desc, "BloomBlur2h");

			desc.ScaleX = 1. / 16;
			desc.ScaleY = 1. / 16;
			m_FrameBuffer_Bloom[7] = FrameBuffer::Create(desc, "BloomBlur3v");
			m_FrameBuffer_Bloom[8] = FrameBuffer::Create(desc, "BloomBlur3h");

			desc.ScaleX = 1. / 32;
			desc.ScaleY = 1. / 32;
			m_FrameBuffer_Bloom[9] = FrameBuffer::Create(desc, "BloomBlur4v");
			m_FrameBuffer_Bloom[10] = FrameBuffer::Create(desc, "BloomBlur4h");
		}

		void SRPPipeBloom::Attach()
		{
			BloomExtractInstance->SetFloat("uBloomThreshold", 2);
			BloomExtractInstance->SetRenderTarget2D("ExtractResult", m_FrameBuffer_Bloom[0], 0);

			BlurLevel0InstanceV->SetFloat2("uBlurDir", { 0,1 });
			BlurLevel0InstanceV->SetTexture2D("u_Input", m_FrameBuffer_Bloom[0]->GetRenderTarget(0));
			BlurLevel0InstanceV->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[1], 0);

			BlurLevel0InstanceH->SetFloat2("uBlurDir", { 1,0 });
			BlurLevel0InstanceH->SetTexture2D("u_Input", m_FrameBuffer_Bloom[1]->GetRenderTarget(0));
			BlurLevel0InstanceH->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[2], 0);

			BlurLevel1InstanceV->SetFloat2("uBlurDir", { 0,1 });
			BlurLevel1InstanceV->SetTexture2D("u_Input", m_FrameBuffer_Bloom[2]->GetRenderTarget(0));
			BlurLevel1InstanceV->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[3], 0);

			BlurLevel1InstanceH->SetFloat2("uBlurDir", { 1,0 });
			BlurLevel1InstanceH->SetTexture2D("u_Input", m_FrameBuffer_Bloom[3]->GetRenderTarget(0));
			BlurLevel1InstanceH->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[4], 0);


			BlurLevel2InstanceV->SetFloat2("uBlurDir", { 0,1 });
			BlurLevel2InstanceV->SetTexture2D("u_Input", m_FrameBuffer_Bloom[4]->GetRenderTarget(0));
			BlurLevel2InstanceV->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[5], 0);

			BlurLevel2InstanceH->SetFloat2("uBlurDir", { 1,0 });
			BlurLevel2InstanceH->SetTexture2D("u_Input", m_FrameBuffer_Bloom[5]->GetRenderTarget(0));
			BlurLevel2InstanceH->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[6], 0);


			BlurLevel3InstanceV->SetFloat2("uBlurDir", { 0,1 });
			BlurLevel3InstanceV->SetTexture2D("u_Input", m_FrameBuffer_Bloom[6]->GetRenderTarget(0));
			BlurLevel3InstanceV->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[7], 0);

			BlurLevel3InstanceH->SetFloat2("uBlurDir", { 1,0 });
			BlurLevel3InstanceH->SetTexture2D("u_Input", m_FrameBuffer_Bloom[7]->GetRenderTarget(0));
			BlurLevel3InstanceH->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[8], 0);


			BlurLevel4InstanceV->SetFloat2("uBlurDir", { 0,1 });
			BlurLevel4InstanceV->SetTexture2D("u_Input", m_FrameBuffer_Bloom[8]->GetRenderTarget(0));
			BlurLevel4InstanceV->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[9], 0);

			BlurLevel4InstanceH->SetFloat2("uBlurDir", { 1,0 });
			BlurLevel4InstanceH->SetTexture2D("u_Input", m_FrameBuffer_Bloom[9]->GetRenderTarget(0));
			BlurLevel4InstanceH->SetRenderTarget2D("BlurResult", m_FrameBuffer_Bloom[10], 0);

			BloomCombineInstance->SetFloat("uBloomFactor", 0.2175);
			BloomCombineInstance->SetTexture2D("TextureBloomBlur1", m_FrameBuffer_Bloom[2]->GetRenderTarget(0));
			BloomCombineInstance->SetTexture2D("TextureBloomBlur2", m_FrameBuffer_Bloom[4]->GetRenderTarget(0));
			BloomCombineInstance->SetTexture2D("TextureBloomBlur3", m_FrameBuffer_Bloom[6]->GetRenderTarget(0));
			BloomCombineInstance->SetTexture2D("TextureBloomBlur4", m_FrameBuffer_Bloom[8]->GetRenderTarget(0));
			BloomCombineInstance->SetTexture2D("TextureBloomBlur5", m_FrameBuffer_Bloom[10]->GetRenderTarget(0));
			BloomCombineInstance->SetRenderTarget2D("CombineResult", m_FrameBuffer_Bloom[11], 0);
		}

		void SRPPipeBloom::Draw()
		{
			auto [screenX, screenY] = SRenderContext::GetScreenSize();
			BloomExtractInstance->SetFloat2("OutputSize", { screenX, screenY });
			BloomExtractInstance->Dispatch(GRIDSIZE(screenX, 16), GRIDSIZE(screenY, 16), 1);

			BlurLevel0InstanceV->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel0InstanceV->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 1, screenY * 1. / 1 });
			BlurLevel0InstanceV->SetFloat2("OutputSize", { screenX * 1. / 2, screenY * 1. / 2 });
			BlurLevel0InstanceV->Dispatch(GRIDSIZE(screenX * 1. / 2, 16), GRIDSIZE(screenY * 1. / 2, 16), 1);

			BlurLevel0InstanceH->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel0InstanceH->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 2, screenY * 1. / 2 });
			BlurLevel0InstanceH->SetFloat2("OutputSize", { screenX * 1. / 2, screenY * 1. / 2 });
			BlurLevel0InstanceH->Dispatch(GRIDSIZE(screenX * 1. / 2, 16), GRIDSIZE(screenY * 1. / 2, 16), 1);


			BlurLevel1InstanceV->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel1InstanceV->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 2, screenY * 1. / 2 });
			BlurLevel1InstanceV->SetFloat2("OutputSize", { screenX * 1. / 4, screenY * 1. / 4 });
			BlurLevel1InstanceV->Dispatch(GRIDSIZE(screenX * 1. / 4, 16), GRIDSIZE(screenY * 1. / 4, 16), 1);

			BlurLevel1InstanceH->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel1InstanceH->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 4, screenY * 1. / 4 });
			BlurLevel1InstanceH->SetFloat2("OutputSize", { screenX * 1. / 4, screenY * 1. / 4 });
			BlurLevel1InstanceH->Dispatch(GRIDSIZE(screenX * 1. / 4, 16), GRIDSIZE(screenY * 1. / 4, 16), 1);

			BlurLevel2InstanceV->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel2InstanceV->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 4, screenY * 1. / 4 });
			BlurLevel2InstanceV->SetFloat2("OutputSize", { screenX * 1. / 8, screenY * 1. / 8 });
			BlurLevel2InstanceV->Dispatch(GRIDSIZE(screenX * 1. / 8, 16), GRIDSIZE(screenY * 1. / 8, 16), 1);

			BlurLevel2InstanceH->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel2InstanceH->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 8, screenY * 1. / 8 });
			BlurLevel2InstanceH->SetFloat2("OutputSize", { screenX * 1. / 8, screenY * 1. / 8 });
			BlurLevel2InstanceH->Dispatch(GRIDSIZE(screenX * 1. / 8, 16), GRIDSIZE(screenY * 1. / 8, 16), 1);

			BlurLevel3InstanceV->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel3InstanceV->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 8, screenY * 1. / 8 });
			BlurLevel3InstanceV->SetFloat2("OutputSize", { screenX * 1. / 16, screenY * 1. / 16 });
			BlurLevel3InstanceV->Dispatch(GRIDSIZE(screenX * 1. / 16, 16), GRIDSIZE(screenY * 1. / 16, 16), 1);

			BlurLevel3InstanceH->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel3InstanceH->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 16, screenY * 1. / 16 });
			BlurLevel3InstanceH->SetFloat2("OutputSize", { screenX * 1. / 16, screenY * 1. / 16 });
			BlurLevel3InstanceH->Dispatch(GRIDSIZE(screenX * 1. / 16, 16), GRIDSIZE(screenY * 1. / 16, 16), 1);

			BlurLevel4InstanceV->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel4InstanceV->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 16, screenY * 1. / 16 });
			BlurLevel4InstanceV->SetFloat2("OutputSize", { screenX * 1. / 32, screenY * 1. / 32 });
			BlurLevel4InstanceV->Dispatch(GRIDSIZE(screenX * 1. / 32, 16), GRIDSIZE(screenY * 1. / 32, 16), 1);

			BlurLevel4InstanceH->SetFloat2("uGlobalTexSize", { screenX,screenY });
			BlurLevel4InstanceH->SetFloat2("uTextureBlurInputSize", { screenX * 1. / 32, screenY * 1. / 32 });
			BlurLevel4InstanceH->SetFloat2("OutputSize", { screenX * 1. / 32, screenY * 1. / 32 });
			BlurLevel4InstanceH->Dispatch(GRIDSIZE(screenX * 1. / 32, 16), GRIDSIZE(screenY * 1. / 32, 16), 1);

			BloomCombineInstance->SetFloat2("OutputSize", { screenX, screenY });
			BloomCombineInstance->Dispatch(GRIDSIZE(screenX, 16), GRIDSIZE(screenY, 16), 1);
		}

		void SRPPipeBloom::DrawImGui()
		{

		}

		RenderTarget* SRPPipeBloom::GetRenderTarget(const std::string& name)
		{
			if (name == "Output")
			{
				return m_FrameBuffer_Bloom[11]->GetRenderTarget(0);
			}

			SIByL_CORE_ERROR("TAA Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipeBloom::SetInput(const std::string& name, RenderTarget* target)
		{
			if (name == "Input")
			{
				BloomExtractInstance->SetTexture2D("u_Texture", target);
				BloomCombineInstance->SetTexture2D("u_Texture", target);
			}
			else if (name == "Depth")
			{
				BloomExtractInstance->SetTexture2D("u_Depth", target);
			}
			else
			{
				SIByL_CORE_ERROR("TAA Pipe get wrong input set");
			}
		}

	}
}