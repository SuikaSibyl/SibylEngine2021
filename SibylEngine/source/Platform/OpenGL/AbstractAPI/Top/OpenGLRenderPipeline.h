#pragma once

namespace SIByL
{
	class OpenGLRenderPipeline
	{
	public:
		OpenGLRenderPipeline();
		void static DrawFrame() { Main->DrawFrameImpl(); }

	protected:
		void DrawFrameImpl();
		static OpenGLRenderPipeline* Main;
	};
}