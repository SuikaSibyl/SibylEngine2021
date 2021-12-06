#pragma once

namespace SIByL
{
	class Scene;
	class Camera;

	namespace SRenderPipeline
	{
		class SPipeline;
		class SRenderContext
		{
		public:
			using pair_uint = std::pair<unsigned int, unsigned int>;
			static pair_uint GetScreenSize() { return pair_uint{ ScreenWidth, ScreenHeight }; }
			static void SetScreenSize(pair_uint size) { ScreenWidth = size.first; ScreenHeight = size.second; }

			static void SetActiveScene(Ref<Scene> scene) { ActiveScene = scene; }
			static Ref<Scene> GetActiveScene() { return ActiveScene; }

			static void SetActiveRenderPipeline(Ref<SPipeline> rp) { ActiveRP = rp; }
			static Ref<SPipeline> GetRenderPipeline() { return ActiveRP; }

			static void SetActiveCamera(Ref<Camera> ca) { ActiveCamera = ca; }
			static Ref<Camera> GetCamera() { return ActiveCamera; }

			static float GetDelta();

		private:
			static unsigned int ScreenWidth, ScreenHeight;
			static Ref<Scene> ActiveScene;
			static Ref<SPipeline> ActiveRP;
			static Ref<Camera> ActiveCamera;
		};

	}
}