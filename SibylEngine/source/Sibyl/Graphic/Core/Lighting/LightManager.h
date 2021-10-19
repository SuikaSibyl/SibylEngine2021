#pragma once

namespace SIByL
{
	class LightComponent;
	class FrameConstantsManager;
	class LightManager
	{
	public:
		static void AddLight(LightComponent*);
		static void RemoveLight(LightComponent*);
		static void SetFrameConstantsManager(FrameConstantsManager* m);
		static void OnUpdate();

	private:
		static std::vector<LightComponent*> Lights;
		static unsigned int DirectionalLightCount;
		static unsigned int PointLightCount;
		static FrameConstantsManager* frameConstantsManager;
	};
}