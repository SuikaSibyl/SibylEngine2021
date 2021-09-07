#include "SIByLpch.h"
#include "DefaultRenderPipeline.h"

#include "ScriptableRenderContext.h"
#include "CommandBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"

#include "Sibyl/Graphic/Core/Material/Color.h"
#include "Culling.h"
#include "Drawing.h"

namespace SIByL
{
	class CameraRenderer
	{
		ScriptableRenderContext* context;
		Camera* camera;
		CommandBuffer buffer;
		std::string bufferName = "Render Camera";

	public:
		CameraRenderer()
			: buffer(bufferName)
			, context(nullptr)
			, camera(nullptr)
		{

		}

		void Render(ScriptableRenderContext* context, Camera* camera) {
			this->context = context;
			this->camera = camera;

			Setup();
			DrawVisibleGeometry();
			Submit();

		}

		void Setup() {
			context->SetupCameraProperties(camera);
			buffer.ClearRenderTarget(true, true, Color(glm::vec4{ 0,0,0,0 }), 1.0f);
			buffer.BeginSample(bufferName);
		}

		void DrawVisibleGeometry() {
			SortingSettings sortingSettings(camera);
			sortingSettings.criteria = SortingCriteria::CommonOpaque;
			DrawingSettings drawingSettings(0, sortingSettings);
			FilteringSettings filteringSettings(RenderQueueRange::all);

			context->DrawRenderers(
				cullingResults, drawingSettings, filteringSettings
			);

			buffer.EndSample(bufferName);
			context->DrawSkybox(camera);
		}

		void Submit() {
			context->Submit();
		}

		void ExecuteBuffer() {
			context->ExecuteCommandBuffer(&buffer);
			buffer.Clear();
		}

		bool Cull() {
			ScriptableCullingParameters p;
			if (camera->TryGetCullingParameters(p)) {
				cullingResults = context->Cull(p);
				return true;
			}
			return false;
		}

		CullingResults cullingResults;
	};

	CameraRenderer renderer;

	void DefaultRenderPipeline::Render(ScriptableRenderContext* context, std::vector<Camera*> cameras)
	{
		for each (Camera* camera in cameras)
		{
			renderer.Render(context, camera);
		}
	}
}