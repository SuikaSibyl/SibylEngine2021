#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Top/RenderPipeline.h"

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