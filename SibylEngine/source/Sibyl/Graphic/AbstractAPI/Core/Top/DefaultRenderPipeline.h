#pragma once

#include "RenderPipeline.h"

namespace SIByL
{
	class DefaultRenderPipeline :public RenderPipeline
	{
	protected:
		virtual void Render(ScriptableRenderContext* context, std::vector<Camera*> cameras) override;

	};

}