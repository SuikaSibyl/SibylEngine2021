#pragma once

namespace SIByL
{
	class ShaderConstantsBuffer;
	class FrameConstantsManager
	{
	public:
		FrameConstantsManager();
		void OnDrawCall();
		void SetFrame();

	private:
		Ref<ShaderConstantsBuffer> m_ConstantsBuffer = nullptr;

	};
}