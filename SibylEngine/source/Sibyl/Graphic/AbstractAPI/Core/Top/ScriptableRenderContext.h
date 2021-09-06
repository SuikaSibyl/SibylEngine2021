#pragma once

namespace SIByL
{
	class Camera;
	class CommandBuffer;

	class ScriptableRenderContext
	{
	public:
		void Submit();
		void DrawSkybox(Camera* camera);
		void SetupCameraProperties(Camera* camera);
		void ExecuteCommandBuffer(CommandBuffer* buffer);
	};
}