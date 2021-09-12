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
	//Assimp::Importer importer;
	//const aiScene* scene = importer.ReadFile("Mesh/LN2_VisDev.fbx", aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenNormals);

	//_CrtSetBreakAlloc(7833);
	Renderer::SetRaster(SIByL::RasterRenderer::DirectX12);
	Renderer::SetRayTracer(SIByL::RayTracerRenderer::Cuda);
	return new Editor();
}