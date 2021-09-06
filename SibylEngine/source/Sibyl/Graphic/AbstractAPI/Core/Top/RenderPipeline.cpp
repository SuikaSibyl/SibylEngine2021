#include "SIByLpch.h"
#include "RenderPipeline.h"

#include "ScriptableRenderContext.h"
#include "DefaultRenderPipeline.h"

namespace SIByL
{
	RenderPipeline* RenderPipeline::ThePipeline = nullptr;
	DefaultRenderPipeline defaultPipeline;

	void RenderPipeline::Set(RenderPipeline* pipeline)
	{

	}

	RenderPipeline* RenderPipeline::Get()
	{
		if (ThePipeline == nullptr)
			ThePipeline = &defaultPipeline;

		return ThePipeline;
	}
}