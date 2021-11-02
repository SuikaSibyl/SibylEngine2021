#pragma once

#include "SIByLpch.h"
#include "FXAA.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/ComputeInstance.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"


namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipeFXAA::Build()
		{
			// Create Compute Shader
			FXAAInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\FXAA"));

			// FrameBufferDesc desc;
			FrameBufferDesc desc;
			desc.Width = 1280;
			desc.Height = 720;
			desc.Formats = { FrameBufferTextureFormat::RGB8 };

			m_FrameBuffer_FXAA = FrameBuffer::Create(desc, "FXAA");
		}

		void SRPPipeFXAA::Attach()
		{
			FXAAInstance->SetRenderTarget2D("FXAAResult", m_FrameBuffer_FXAA, 0);
		}

		void SRPPipeFXAA::Draw()
		{
			auto& [screenX, screenY] = SRenderContext::GetScreenSize();
			FXAAInstance->SetFloat2("OutputSize", { screenX, screenY });
			FXAAInstance->Dispatch(GRIDSIZE(screenX, 16), GRIDSIZE(screenY, 16), 1);

		}

		void SRPPipeFXAA::DrawImGui()
		{

		}

		RenderTarget* SRPPipeFXAA::GetRenderTarget(const std::string& name)
		{
			if (name == "Output")
			{
				return m_FrameBuffer_FXAA->GetRenderTarget(0);
			}

			SIByL_CORE_ERROR("TAA Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipeFXAA::SetInput(const std::string& name, RenderTarget* target)
		{
			if (name == "Input")
			{
				FXAAInstance->SetTexture2D("u_Texture", target);
			}
			else
			{
				SIByL_CORE_ERROR("TAA Pipe get wrong input set");
			}
		}

	}
}