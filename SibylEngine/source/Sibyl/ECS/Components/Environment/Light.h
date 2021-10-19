#pragma once

#include <Sibyl/Graphic/Core/Lighting/Light.h>

namespace SIByL
{
	struct LightComponent
	{
		LightComponent() = default;

		LightType m_Type = LightType::Directional;
		glm::vec3 m_Position;
		glm::vec3 m_Direction;
		glm::vec3 m_Color;
		float m_Intensity = 1;
		bool isDirty = false;
	};
}