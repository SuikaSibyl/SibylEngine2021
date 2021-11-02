#pragma once

#include "SIByLpch.h"
#include "SSAO.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/ComputeInstance.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"


namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipeSSAO::Build()
		{
			// Create Compute Shader
			SSAOExtractInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\SSAO"));
			MedianBlurVInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\MedianBlurV"));
			MedianBlurHInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\MedianBlurH"));
			SSAOCombineInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\SSAOCombine"));

			// FrameBufferDesc desc;
			FrameBufferDesc desc;
			desc.Width = 1280;
			desc.Height = 720;
			desc.Formats = { FrameBufferTextureFormat::RGB8 };

			m_FrameBuffer_SSAO[0] = FrameBuffer::Create(desc, "SSAO1");
			m_FrameBuffer_SSAO[1] = FrameBuffer::Create(desc, "SSAO2");
			m_FrameBuffer_SSAO[2] = FrameBuffer::Create(desc, "SSAO3");
			m_FrameBuffer_SSAO[3] = FrameBuffer::Create(desc, "SSAO4");
		}

		void SRPPipeSSAO::Attach()
		{
			SSAOExtractInstance->SetFloat("uRadius", 0.01);
			SSAOExtractInstance->SetRenderTarget2D("SSAOResult", m_FrameBuffer_SSAO[0], 0);

			MedianBlurVInstance->SetFloat("u_radius", 1);
			MedianBlurVInstance->SetTexture2D("u_Input", m_FrameBuffer_SSAO[0]->GetRenderTarget(0));
			MedianBlurVInstance->SetRenderTarget2D("MedianBlurResult", m_FrameBuffer_SSAO[1], 0);

			MedianBlurHInstance->SetFloat("u_radius", 1);
			MedianBlurHInstance->SetTexture2D("u_Input", m_FrameBuffer_SSAO[1]->GetRenderTarget(0));
			MedianBlurHInstance->SetRenderTarget2D("MedianBlurResult", m_FrameBuffer_SSAO[2], 0);

			SSAOCombineInstance->SetFloat("uAOFactor", 1.5);
			SSAOCombineInstance->SetTexture2D("u_SSAO", m_FrameBuffer_SSAO[2]->GetRenderTarget(0));
			SSAOCombineInstance->SetRenderTarget2D("SSAOCombinedResult", m_FrameBuffer_SSAO[3], 0);
		}

		void SRPPipeSSAO::Draw()
		{
			const glm::mat4& projection = SRenderContext::GetCamera()->GetProjectionMatrix();
			SSAOExtractInstance->SetMatrix4x4("uProjection", projection);

			auto& [screenX, screenY] = SRenderContext::GetScreenSize();
			SSAOExtractInstance->SetFloat2("OutputSize", { screenX, screenY });
			SSAOExtractInstance->Dispatch(GRIDSIZE(screenX, 16), GRIDSIZE(screenY, 16), 1);

			MedianBlurVInstance->SetFloat2("OutputSize", { screenX, screenY });
			MedianBlurVInstance->Dispatch(GRIDSIZE(screenX, 64), GRIDSIZE(screenY, 1), 1);

			MedianBlurHInstance->SetFloat2("OutputSize", { screenX, screenY });
			MedianBlurHInstance->Dispatch(GRIDSIZE(screenX, 1), GRIDSIZE(screenY, 64), 1);

			SSAOCombineInstance->SetFloat2("OutputSize", { screenX, screenY });
			SSAOCombineInstance->Dispatch(GRIDSIZE(screenX, 16), GRIDSIZE(screenY, 16), 1);
		}

		void SRPPipeSSAO::DrawImGui()
		{

		}

		RenderTarget* SRPPipeSSAO::GetRenderTarget(const std::string& name)
		{
			if (name == "Output")
			{
				return m_FrameBuffer_SSAO[3]->GetRenderTarget(0);
			}

			SIByL_CORE_ERROR("TAA Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipeSSAO::SetInput(const std::string& name, RenderTarget* target)
		{
			if (name == "Color")
			{
				SSAOCombineInstance->SetTexture2D("u_Texture", target);
			}
			else if (name == "Normal")
			{
				SSAOExtractInstance->SetTexture2D("u_Normal", target);
			}
			else if (name == "Depth")
			{
				SSAOExtractInstance->SetTexture2D("u_Depth", target);
			}
			else
			{
				SIByL_CORE_ERROR("TAA Pipe get wrong input set");
			}
		}

	}
}