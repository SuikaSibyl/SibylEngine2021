#pragma once

namespace SIByL
{
	class Camera;
	class ScriptableRenderContext;

	class RenderPipeline
	{
	public:
		virtual ~RenderPipeline() = default;
		static void Set(RenderPipeline* pipeline);
		static RenderPipeline* Get();
		virtual void Render(ScriptableRenderContext* context, std::vector<Camera*> cameras) = 0;

	protected:

		static RenderPipeline* ThePipeline;
	};
}