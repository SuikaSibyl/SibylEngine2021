#pragma once

#include "SIByLpch.h"
#include "Sharpen.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Top/ComputeInstance.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"


namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipeSharpen::Build()
		{
			// Create Compute Shader
			SharpenInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\FXAA"));

			// FrameBufferDesc desc;
			FrameBufferDesc desc;
			desc.Width = 1280;
			desc.Height = 720;
			desc.Formats = { FrameBufferTextureFormat::RGB8 };

			m_FrameBuffer_Sharpen = FrameBuffer::Create(desc, "Sharpen");
		}

		void SRPPipeSharpen::Attach()
		{
			SharpenInstance->SetFloat("uSharpFactor", 1);
			SharpenInstance->SetRenderTarget2D("FXAAResult", m_FrameBuffer_Sharpen, 0);
		}

		void SRPPipeSharpen::Draw()
		{
			auto& [screenX, screenY] = SRenderContext::GetScreenSize();
			SharpenInstance->Dispatch(screenX, screenY, 1);
		}

		void SRPPipeSharpen::DrawImGui()
		{

		}

		RenderTarget* SRPPipeSharpen::GetRenderTarget(const std::string& name)
		{
			if (name == "Output")
			{
				return m_FrameBuffer_Sharpen->GetRenderTarget(0);
			}

			SIByL_CORE_ERROR("TAA Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipeSharpen::SetInput(const std::string& name, RenderTarget* target)
		{
			if (name == "Input")
			{
				SharpenInstance->SetTexture2D("u_Texture", target);
			}
			else if (name == "Depth")
			{
				SharpenInstance->SetTexture2D("u_Depth", target);
			}
			else
			{
				SIByL_CORE_ERROR("TAA Pipe get wrong input set");
			}
		}

	}
}