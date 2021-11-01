#pragma once

#include "SIByLpch.h"
#include "Vignette.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/ComputeInstance.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"


namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipeVignette::Build()
		{
			// Create Compute Shader
			VignetteInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\Vignette"));

			// FrameBufferDesc desc;
			FrameBufferDesc desc;
			desc.Width = 1280;
			desc.Height = 720;
			desc.Formats = { FrameBufferTextureFormat::RGB8 };

			m_FrameBuffer_Vignette = FrameBuffer::Create(desc, "Vignette");
		}

		void SRPPipeVignette::Attach()
		{
			VignetteInstance->SetFloat2("uLensRadius", { 0.8000, 0.2500 });
			VignetteInstance->SetFloat("uFrameMod", 7);
			VignetteInstance->SetRenderTarget2D("VignetteResult", m_FrameBuffer_Vignette, 0);
		}

		void SRPPipeVignette::Draw()
		{
			auto& [screenX, screenY] = SRenderContext::GetScreenSize();
			VignetteInstance->Dispatch(screenX, screenY, 1);
		}

		void SRPPipeVignette::DrawImGui()
		{

		}

		RenderTarget* SRPPipeVignette::GetRenderTarget(const std::string& name)
		{
			if (name == "Output")
			{
				return m_FrameBuffer_Vignette->GetRenderTarget(0);
			}

			SIByL_CORE_ERROR("TAA Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipeVignette::SetInput(const std::string& name, RenderTarget* target)
		{
			if (name == "Input")
			{
				VignetteInstance->SetTexture2D("u_Texture", target);
			}
			else
			{
				SIByL_CORE_ERROR("TAA Pipe get wrong input set");
			}
		}

	}
}