#include "SIByLpch.h"
#include "LightManager.h"

#include <Sibyl/ECS/Components/Environment/Light.h>
#include "Sibyl/Graphic/AbstractAPI/Core/Top/FrameConstantsManager.h"
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h>

namespace SIByL
{
	std::vector<LightComponent*> LightManager::Lights;
	unsigned int LightManager::DirectionalLightCount = 0;
	unsigned int LightManager::PointLightCount = 0;
	FrameConstantsManager* LightManager::frameConstantsManager = nullptr;

	void LightManager::AddLight(LightComponent* light)
	{
		Lights.push_back(light);
	}

	void LightManager::RemoveLight(LightComponent* light)
	{
		for (auto iter = Lights.begin(); iter != Lights.end(); iter++)
		{
			if (*iter == light)
			{
				Lights.erase(iter);
				return;
			}
		}
	}

	void LightManager::SetFrameConstantsManager(FrameConstantsManager* m)
	{
		frameConstantsManager = m;
	}

	void LightManager::OnUpdate()
	{
		DirectionalLightCount = 0;
		PointLightCount = 0;

		ShaderConstantsBuffer* frameBuffer = frameConstantsManager->GetShaderConstantsBuffer();

		for (auto iter = Lights.begin(); iter != Lights.end(); iter++)
		{
			if ((*iter)->m_Type == LightType::Directional)
			{
				frameBuffer->SetMatrix4x4("directionalLights[" + std::to_string(DirectionalLightCount) + "].projview", (*iter)->m_LightProjView);
				frameBuffer->SetFloat3("directionalLights[" + std::to_string(DirectionalLightCount) + "].direction", (*iter)->m_Direction);
				frameBuffer->SetFloat3("directionalLights[" + std::to_string(DirectionalLightCount) + "].color", (*iter)->m_Color);
				frameBuffer->SetFloat("directionalLights[" + std::to_string(DirectionalLightCount) + "].intensity", 1);
				DirectionalLightCount++;
			}
			else if ((*iter)->m_Type == LightType::Point)
			{

			}
		}
		frameBuffer->SetInt("DirectionalLightNum", DirectionalLightCount);

	}

}