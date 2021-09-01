#pragma once

#include "Sibyl/Graphic/AbstractAPI/Top/RenderPipeline.h"

namespace SIByL
{
	class OpenGLRenderPipeline :public RenderPipeline
	{
	public:
		OpenGLRenderPipeline();
		void static DrawFrame() { Main->DrawFrameImpl(); }

	protected:
		void DrawFrameImpl();
		static OpenGLRenderPipeline* Main;
	};
}