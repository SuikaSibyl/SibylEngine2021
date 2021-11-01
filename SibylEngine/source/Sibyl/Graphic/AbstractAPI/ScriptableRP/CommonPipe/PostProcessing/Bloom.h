#pragma once
#pragma once

#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/SPipe.h"

namespace SIByL
{
	class ComputeInstance;

	namespace SRenderPipeline
	{
		class SRPPipeBloom :public SPipePostProcess
		{
		public:
			SPipeBegin(SRPPipeBloom)
				virtual void Build() override;
			virtual void Attach() override;
			virtual void Draw() override;
			virtual void DrawImGui()override;

			virtual RenderTarget* GetRenderTarget(const std::string& name) override;
			virtual void SetInput(const std::string& name, RenderTarget* target) override;

		private:
			Ref<ComputeInstance> BloomExtractInstance;
			Ref<ComputeInstance> BloomCombineInstance;
			Ref<ComputeInstance> BlurLevel0InstanceV;
			Ref<ComputeInstance> BlurLevel0InstanceH;
			Ref<ComputeInstance> BlurLevel1InstanceV;
			Ref<ComputeInstance> BlurLevel1InstanceH;
			Ref<ComputeInstance> BlurLevel2InstanceV;
			Ref<ComputeInstance> BlurLevel2InstanceH;
			Ref<ComputeInstance> BlurLevel3InstanceV;
			Ref<ComputeInstance> BlurLevel3InstanceH;
			Ref<ComputeInstance> BlurLevel4InstanceV;
			Ref<ComputeInstance> BlurLevel4InstanceH;

			Ref<FrameBuffer> m_FrameBuffer_Bloom[12];

			int mTaaBufferIdx = 0;
		};
	}
}