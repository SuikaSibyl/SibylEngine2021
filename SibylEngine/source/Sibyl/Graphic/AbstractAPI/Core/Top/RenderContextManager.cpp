#include "SIByLpch.h"
#include "RenderContextManager.h"

#include "ScriptableRenderContext.h"
#include "Camera.h"

namespace SIByL
{
	RenderContext* RenderContext::instance = nullptr;

	RenderContext::RenderContext()
	{
		m_ScriptableRenderContext = CreateScope<ScriptableRenderContext>();
	}

	RenderContext* RenderContext::Get()
	{
		if (!instance) instance = new RenderContext();

		return instance;
	}

	ScriptableRenderContext* RenderContext::GetSRContext()
	{
		return Get()->m_ScriptableRenderContext.get();
	}

	std::vector<Camera*> RenderContext::GetCameras()
	{
		return Get()->m_Cameras;
	}

	void RenderContext::Reset()
	{
		delete instance;
	}

}