#pragma once
#pragma once

#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SPipe.h"

namespace SIByL
{
	class ComputeInstance;

	namespace SRenderPipeline
	{
		class SRPPipeSSAO :public SPipePostProcess
		{
		public:
			SPipeBegin(SRPPipeSSAO)
			virtual void Build() override;
			virtual void Attach() override;
			virtual void Draw() override;
			virtual void DrawImGui()override;

			virtual RenderTarget* GetRenderTarget(const std::string& name) override;
			virtual void SetInput(const std::string& name, RenderTarget* target) override;

		private:
			Ref<ComputeInstance> SSAOExtractInstance;
			Ref<ComputeInstance> MedianBlurVInstance;
			Ref<ComputeInstance> MedianBlurHInstance;
			Ref<ComputeInstance> SSAOCombineInstance;
			Ref<FrameBuffer> m_FrameBuffer_SSAO[4];
		};
	}
}