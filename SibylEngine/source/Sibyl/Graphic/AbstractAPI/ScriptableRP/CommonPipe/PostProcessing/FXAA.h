#pragma once

#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SPipe.h"

namespace SIByL
{
	class ComputeInstance;

	namespace SRenderPipeline
	{
		class SRPPipeFXAA :public SPipePostProcess
		{
		public:
			SPipeBegin(SRPPipeFXAA)

			virtual void Build() override;
			virtual void Attach() override;
			virtual void Draw() override;
			virtual void DrawImGui()override;

			virtual RenderTarget* GetRenderTarget(const std::string& name) override;
			virtual void SetInput(const std::string& name, RenderTarget* target) override;

		private:
			Ref<ComputeInstance> FXAAInstance;
			Ref<FrameBuffer> m_FrameBuffer_FXAA;
		};
	}
}