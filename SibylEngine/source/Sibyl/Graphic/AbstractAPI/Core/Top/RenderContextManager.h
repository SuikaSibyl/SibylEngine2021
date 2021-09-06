#pragma once

namespace SIByL
{
	class Camera;
	class ScriptableRenderContext;

	class RenderContext
	{
	public:
		RenderContext();
		static RenderContext* Get();
		static void Reset();
		static ScriptableRenderContext* GetSRContext();
		static std::vector<Camera*> GetCameras();

	private:
		Scope<ScriptableRenderContext> m_ScriptableRenderContext;
		std::vector<Camera*> m_Cameras;
		static RenderContext* instance;
	};
}