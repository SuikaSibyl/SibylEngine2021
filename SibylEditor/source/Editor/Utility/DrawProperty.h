#pragma once

#include "glm/glm.hpp"
#include "imgui.h"
#include "imgui_internal.h"

#include "Sibyl/ECS/Components/Components.h"

namespace SIByL
{
	class Material;
	class Texture2D;
	class TriangleMesh;
	class ShaderConstantItem;
}

namespace SIByLEditor
{
	void DrawVec3Control(const std::string& label, glm::vec3& values, float resetValue = 0.0f, float columeWidth = 100);
	void DrawTriangleMeshSocket(const std::string& label, SIByL::MeshFilterComponent& mesh);

	void DrawTexture2D(SIByL::Material& material, SIByL::ShaderConstantItem& item);

	void DrawMaterial(const std::string& label, SIByL::Material& material);
}