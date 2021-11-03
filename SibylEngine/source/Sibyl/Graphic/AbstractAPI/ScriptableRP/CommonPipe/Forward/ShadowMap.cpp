#include "SIByLpch.h"
#include "ShadowMap.h"

#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"
#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SRenderContext.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"
#include "Sibyl/Graphic/Core/Lighting/LightManager.h"
#include "Sibyl/ECS/Scene/Scene.h"
#include <Sibyl/ECS/Components/Environment/Light.h>
#include "Sibyl/Graphic/AbstractAPI/Core/Top/FrameConstantsManager.h"

namespace SIByL
{
	namespace SRenderPipeline
	{
		void SRPPipeShadowMap::Build()
		{
			FrameBufferDesc desc;
			desc.Width = 2048;
			desc.Height = 2048;
			desc.Formats = {
				FrameBufferTextureFormat::RGB8,		// Shadowmap
				FrameBufferTextureFormat::DEPTH24STENCIL8 };
			desc.ScaleX = -1;
			desc.ScaleY = -1;
			// Frame Buffer 0: Main Render Buffer
			mDirectionalShadowmap = FrameBuffer::Create(desc, "DirectionalShadowmap");
		}

		void SRPPipeShadowMap::Attach()
		{

		}

		void SRPPipeShadowMap::Draw()
		{
			struct viewport
			{
				unsigned int xmin;
				unsigned int ymin;
				unsigned int xmax;
				unsigned int ymax;
			};
			static viewport shadowViewport[4] =
			{
				{0,0,1024,1024},
				{1024,0,1024,1024},
				{0,1024,1024,1024},
				{1024,1024,1024,1024},
			};

			mDirectionalShadowmap->Bind();
			mDirectionalShadowmap->ClearBuffer();

			std::vector<LightComponent*>& lights = LightManager::GetLights();
			int directionalIdx = 0;
			for (LightComponent* light : lights)
			{
				if (light->m_Type == LightType::Directional && directionalIdx < 4)
				{
					OrthographicCamera camera(1024, 1024);
					camera.SetDirection(light->m_Direction);
					camera.SetPosition(glm::vec3(0));
					camera.Dither(0, 0);
					camera.UpdateZNearFar();
					light->m_LightProjView = camera.GetProjectionViewMatrix();

					mDirectionalShadowmap->CustomViewport(shadowViewport[directionalIdx].xmin, shadowViewport[directionalIdx].ymin,
						shadowViewport[directionalIdx].xmax, shadowViewport[directionalIdx].ymax);

					DrawItemPool& diPool = SRenderContext::GetActiveScene()->GetDrawItems("ShadowMap");
					for (Ref<DrawItem> drawItem : diPool)
					{
						drawItem->m_Material->SetPass();
						camera.OnDrawCall();
						Graphic::CurrentMaterial->OnDrawCall();
						Graphic::CurrentFrameConstantsManager->OnDrawCall();
						drawItem->OnDrawCall();
					}

					directionalIdx++;
				}
			}

			mDirectionalShadowmap->Unbind();
		}

		void SRPPipeShadowMap::DrawImGui()
		{

		}

		RenderTarget* SRPPipeShadowMap::GetRenderTarget(const std::string& name)
		{
			if (name == "DirectionalShadowmap")
			{
				return mDirectionalShadowmap->GetRenderTarget(0);
			}

			SIByL_CORE_ERROR("TAA Pipe get wrong output required");
			return nullptr;
		}

		void SRPPipeShadowMap::SetInput(const std::string& name, RenderTarget* target)
		{

		}
	}
}