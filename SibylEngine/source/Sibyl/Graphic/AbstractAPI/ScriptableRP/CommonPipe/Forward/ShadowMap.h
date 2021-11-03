#pragma once

#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SPipe.h"

namespace SIByL
{
	namespace SRenderPipeline
	{
		class SRPPipeShadowMap :public SPipeDrawPass
		{
		public:
			SPipeBegin(SRPPipeShadowMap)

			virtual void Build() override;
			virtual void Attach() override;
			virtual void Draw() override;
			virtual void DrawImGui() override;

			virtual RenderTarget* GetRenderTarget(const std::string& name) override;
			virtual void SetInput(const std::string& name, RenderTarget* target) override;

		private:
			Ref<FrameBuffer> mDirectionalShadowmap;
		};
	}
}