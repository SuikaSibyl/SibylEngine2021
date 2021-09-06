#include "SIByLpch.h"
#include "GraphicContext.h"

#include "RenderContextManager.h"

namespace SIByL
{
	GraphicContext::~GraphicContext()
	{
		RenderContext::Reset();
	}

}