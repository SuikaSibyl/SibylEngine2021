#include "SIByLpch.h"
#include "ScriptableRenderContext.h"

#include "CommandBuffer.h"
#include "Culling.h"
#include "Drawing.h"

namespace SIByL
{
	void ScriptableRenderContext::Submit()
	{

	}

	void ScriptableRenderContext::DrawSkybox(Camera* camera)
	{

	}

	void ScriptableRenderContext::SetupCameraProperties(Camera* camera)
	{

	}

	void ScriptableRenderContext::ExecuteCommandBuffer(CommandBuffer* buffer)
	{

	}

	CullingResults ScriptableRenderContext::Cull(ScriptableCullingParameters& cullParas)
	{
		return CullingResults{};
	}

	void ScriptableRenderContext::DrawRenderers(CullingResults& cullingResults, 
		DrawingSettings& drawingSettings, FilteringSettings& filteringSettings)
	{

	}

}