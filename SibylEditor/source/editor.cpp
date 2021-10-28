#include <SIByL.h>
#include "Sibyl/Core/EntryPoint.h"
#include "EditorLayer.h"

using namespace SIByL;

class Editor :public SIByL::Application
{
public:
	Editor()
		:Application("SIByL Editor")
	{
		PushLayer(new SIByLEditor::EditorLayer());
	}

	~Editor()
	{

	}
};

SIByL::Application* SIByL::CreateApplication()
{
	glm::mat4x4 ModelMatrix = { 1, -0.0000, 0, 0, 0.0000, 0.0000, 1, 0, -0.0000, -1, 0.0000, 0, -0.0000, -0.0000, 0, 1 };
	glm::mat4x4 ModelViewMatrix = { 0.8059, 0.4345, -0.4022, 0, 0, 0.6793, 0.7339, 0, 0.5921, -0.5914, 0.5474, 0, 0.6356, -65.6342, -152.3783, 1 };
	glm::mat4x4 ViewMatrix = ModelViewMatrix * glm::inverse(ModelMatrix);
	glm::vec4 pointViewspace = { -24.2101, 30.7392, -38.2317, 1 };
	glm::vec4 pointWorldspace = glm::inverse(ViewMatrix) * pointViewspace;
	std::cout << pointWorldspace[0] << ", " << pointWorldspace[1] << ", " << pointWorldspace[2] << ", " << pointWorldspace[3] << std::endl;

	glm::vec3 directionViewspace = { 0.3281, -0.8582, 0.3948 };
	glm::mat3x3 ViewNormal = glm::mat3x3(ViewMatrix);
	glm::vec3 directionWorldspace = glm::inverse(ViewNormal) * directionViewspace;
	std::cout << directionWorldspace[0] << ", " << directionWorldspace[1] << ", " << directionWorldspace[2] << std::endl;

	//_CrtSetBreakAlloc(7833);
	Renderer::SetRaster(SIByL::RasterRenderer::OpenGL);
	Renderer::SetRayTracer(SIByL::RayTracerRenderer::Cuda);
	return new Editor();
}