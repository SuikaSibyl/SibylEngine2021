#pragma once

#include "SIByLpch.h"
#include "TAA.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/ComputeInstance.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"


namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipeTAA::Build()
		{
			// Create Compute Shader
			TAAInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\TAA"));

			// FrameBufferDesc desc;
			FrameBufferDesc desc;
			desc.Width = 1280;
			desc.Height = 720;
			desc.Formats = { FrameBufferTextureFormat::RGB8 };

			m_FrameBuffer_TAA[0] = FrameBuffer::Create(desc, "TAA1");
			m_FrameBuffer_TAA[1] = FrameBuffer::Create(desc, "TAA2");
		}

		void SRPPipeTAA::Attach()
		{
			TAAInstance->SetRenderTarget2D("TAAResult", m_FrameBuffer_TAA[1], 0);
			TAAInstance->SetTexture2D("u_PreviousFrame", m_FrameBuffer_TAA[0]->GetRenderTarget(0));
			TAAInstance->SetFloat("Alpha", 0.5);
		}

		void SRPPipeTAA::Draw()
		{
			TAAInstance->SetRenderTarget2D("TAAResult", m_FrameBuffer_TAA[mTaaBufferIdx], 0);
			TAAInstance->SetTexture2D("u_PreviousFrame", m_FrameBuffer_TAA[(mTaaBufferIdx + 1) % 2]->GetRenderTarget(0));

			mTaaBufferIdx++; if (mTaaBufferIdx == 2) mTaaBufferIdx = 0;

			auto& [screenX, screenY] = SRenderContext::GetScreenSize();
			TAAInstance->Dispatch(screenX, screenY, 1);
		}

		void SRPPipeTAA::DrawImGui()
		{

		}

		RenderTarget* SRPPipeTAA::GetRenderTarget(const std::string& name)
		{
			if (name == "Output")
			{
				return m_FrameBuffer_TAA[(mTaaBufferIdx + 1) % 2]->GetRenderTarget(0);
			}

			SIByL_CORE_ERROR("TAA Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipeTAA::SetInput(const std::string& name, RenderTarget* target)
		{
			if (name == "MotionVector")
			{
				TAAInstance->SetTexture2D("u_Offset", target);
			}
			else if (name == "Input")
			{
				TAAInstance->SetTexture2D("u_CurrentFrame", target);
			}
			else
			{
				SIByL_CORE_ERROR("TAA Pipe get wrong input set");
			}
		}

	}
}