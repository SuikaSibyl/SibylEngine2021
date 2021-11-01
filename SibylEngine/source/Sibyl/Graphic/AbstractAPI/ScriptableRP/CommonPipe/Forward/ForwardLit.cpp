#include "SIByLpch.h"
#include "ForwardLit.h"

#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"
#include "Sibyl/ECS/Scene/Scene.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/FrameConstantsManager.h"

namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipeForwardLit::Build()
		{
			FrameBufferDesc desc;
			desc.Width = 1280;
			desc.Height = 720;
			desc.Formats = {
				FrameBufferTextureFormat::RGB8,
				FrameBufferTextureFormat::R16G16F,
				FrameBufferTextureFormat::DEPTH24STENCIL8 };
			// Frame Buffer 0: Main Render Buffer
			mFrameBuffer = FrameBuffer::Create(desc, "ForwardLit");
		}

		void SRPPipeForwardLit::Attach()
		{

		}

		void SRPPipeForwardLit::Draw()
		{
			mFrameBuffer->Bind();
			mFrameBuffer->ClearBuffer();

			mCamera->SetCamera();
			mCamera->RecordVPMatrix();

			DrawItemPool& diPool = SRenderContext::GetActiveScene()->GetDrawItems("ForwardLit");
			for (Ref<DrawItem> drawItem : diPool)
			{
				drawItem->m_Material->SetPass();
				Graphic::CurrentCamera->OnDrawCall();
				Graphic::CurrentMaterial->OnDrawCall();
				Graphic::CurrentFrameConstantsManager->OnDrawCall();
				drawItem->OnDrawCall();
			}

			mFrameBuffer->Unbind();
		}

		void SRPPipeForwardLit::DrawImGui()
		{

		}

		RenderTarget* SRPPipeForwardLit::GetRenderTarget(const std::string& name)
		{
			if (name == "Color")
			{
				return mFrameBuffer->GetRenderTarget(0);
			}
			else if (name == "MotionVector")
			{
				return mFrameBuffer->GetRenderTarget(1);
			}

			SIByL_CORE_ERROR("TAA Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipeForwardLit::SetInput(const std::string& name, RenderTarget* target)
		{

		}
	}
}