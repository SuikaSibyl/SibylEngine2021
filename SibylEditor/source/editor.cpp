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
	Renderer::SetRaster(SIByL::RasterRenderer::OpenGL);
	Renderer::SetRayTracer(SIByL::RayTracerRenderer::Cuda);
	return new Editor();
}