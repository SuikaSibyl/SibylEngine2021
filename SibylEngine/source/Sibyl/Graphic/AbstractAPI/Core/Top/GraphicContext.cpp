#include "SIByLpch.h"
#include "GraphicContext.h"

#include "Sibyl/Graphic/AbstractAPI/ScriptableRP/RenderContextManager.h"

namespace SIByL
{
	GraphicContext::~GraphicContext()
	{
		RenderContext::Reset();
	}

}