#pragma once

namespace SIByL
{
	class Camera;
	class CommandBuffer;
	class ScriptableCullingParameters;
	class CullingResults;
	class DrawingSettings;
	class FilteringSettings;

	class ScriptableRenderContext
	{
	public:
		void Submit();
		void DrawSkybox(Camera* camera);
		void SetupCameraProperties(Camera* camera);
		void ExecuteCommandBuffer(CommandBuffer* buffer);
		CullingResults Cull(ScriptableCullingParameters& cullParas);
		void DrawRenderers(CullingResults& cullingResults, DrawingSettings& drawingSettings, FilteringSettings& filteringSettings);
	};
}