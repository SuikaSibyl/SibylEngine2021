#pragma once

#include "SIByLpch.h"
#include "ACES.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/ComputeInstance.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"


namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipeACES::Build()
		{
			// Create Compute Shader
			ACESInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\ACES"));

			// FrameBufferDesc desc;
			FrameBufferDesc desc;
			desc.Width = 1280;
			desc.Height = 720;
			desc.Formats = { FrameBufferTextureFormat::RGB8 };

			m_FrameBuffer_ACES = FrameBuffer::Create(desc, "ACES");
		}

		void SRPPipeACES::Attach()
		{
			ACESInstance->SetRenderTarget2D("ACESResult", m_FrameBuffer_ACES, 0);
			//ACESInstance->SetTexture2D("Input", m_FrameBuffer->GetRenderTarget(0));
			ACESInstance->SetFloat("Para", 0.5);
		}

		void SRPPipeACES::Draw()
		{
			auto& [screenX, screenY] = SRenderContext::GetScreenSize();
			ACESInstance->SetFloat2("OutputSize", { screenX, screenY });
			ACESInstance->Dispatch(GRIDSIZE(screenX, 16), GRIDSIZE(screenY, 16), 1);
		}

		void SRPPipeACES::DrawImGui()
		{

		}

		RenderTarget* SRPPipeACES::GetRenderTarget(const std::string& name)
		{
			if (name == "Output")
			{
				return m_FrameBuffer_ACES->GetRenderTarget(0);
			}

			SIByL_CORE_ERROR("ACES Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipeACES::SetInput(const std::string& name, RenderTarget* target)
		{
			if (name == "Input")
			{
				ACESInstance->SetTexture2D("Input", target);
			}
			else
			{
				SIByL_CORE_ERROR("TAA Pipe get wrong input set");
			}
		}

	}
}