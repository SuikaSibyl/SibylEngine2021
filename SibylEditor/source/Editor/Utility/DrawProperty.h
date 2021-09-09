#pragma once

#include "glm/glm.hpp"
#include "imgui.h"
#include "imgui_internal.h"

namespace SIByL
{
	class Material;
}

namespace SIByLEditor
{
	void DrawVec3Control(const std::string& label, glm::vec3& values, float resetValue = 0.0f, float columeWidth = 100);

	void DrawMaterial(const std::string& label, SIByL::Material& material);
}