//#include <SIByL.h>
//
//#include <Sibyl/Core/EntryPoint.h>
//#include <Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h>
//
//using namespace SIByL;
//
//class EditorLayer :public SIByL::Layer
//{
//public:
//	EditorLayer()
//		:Layer("Example")
//	{
//
//	}
//
//	void OnInitResource() override
//	{
//
//	}
//
//	void OnUpdate() override
//	{
//
//	}
//
//	virtual void OnDrawImGui() 
//	{
//
//	}
//
//	void OnEvent(SIByL::Event& event) override
//	{
//
//	}
//
//	void OnDraw() override
//	{
//
//	}
//};
//
//class SIByLEditor :public SIByL::Application
//{
//public:
//	SIByLEditor()
//	{
//		PushLayer(new EditorLayer());
//	}
//
//	~SIByLEditor()
//	{
//
//	}
//};
//
//SIByL::Application* SIByL::CreateApplication()
//{
//	//_CrtSetBreakAlloc(1330);
//	Renderer::SetRaster(SIByL::RasterRenderer::DirectX12);
//	Renderer::SetRayTracer(SIByL::RayTracerRenderer::Cuda);
//	SIByL_APP_TRACE("Create Application");
//	return new SIByLEditor();
//}

//#include <Core/module.h>
#include <FileModule/module.h>
#include <ShaderModule/module.h>

int main()
{
	//SIByL::File::ModuleTest::Test();
	SIByL::SShader::TestModule();
	//S_CORE_INFO(SIByL::SShader::TestModule();
}
	