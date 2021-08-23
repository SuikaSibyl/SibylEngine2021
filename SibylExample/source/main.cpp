#include <iostream>

#include <SIByL.h>

#include <Sibyl/Renderer/Renderer.h>
#include "Sibyl/Renderer/Shader.h"
#include "Sibyl/Graphic/Geometry/TriangleMesh.h"

using namespace SIByL;

class ExampleLayer :public SIByL::Layer
{
public:
	ExampleLayer()
		:Layer("Example")
	{

	}

	void OnInitRenderer() override
	{
		VertexBufferLayout layout =
		{
			{ShaderDataType::Float3, "POSITION"},
		};

		VertexData vertices[] = {
			0.5f, 0.5f, 0.0f,   // 右上角
			0.5f, -0.5f, 0.0f,  // 右下角
			-0.5f, -0.5f, 0.0f, // 左下角
			-0.5f, 0.5f, 0.0f   // 左上角
		};

		uint32_t indices[] = { // 注意索引从0开始! 
			0, 1, 3, // 第一个三角形
			1, 2, 3  // 第二个三角形
		};

		shader = Shader::Create("Test/basic", "Test/basic");
		shader->CreateBinder(layout);
		triangle = TriangleMesh::Create((float*)vertices, 4 * 3, indices, 6, layout);
	}

	void OnUpdate() override
	{
		//SIByL_APP_INFO("ExampleLayer::Update");
		if (SIByL::Input::IsKeyPressed(SIByL_KEY_TAB))
			SIByL_APP_TRACE("Tab key is pressed!");

		SIByL_CORE_INFO("FPS: {0}, {1} ms", 
			Application::Get().GetFrameTimer()->GetFPS(),
			Application::Get().GetFrameTimer()->GetMsPF());

	}

	void OnEvent(SIByL::Event& event) override
	{
		//SIByL_APP_TRACE("{0}", event);
	}

	void OnDraw() override
	{
		shader->Use();
		triangle->RasterDraw();
	}

	Shader* shader;
	TriangleMesh* triangle;
};

class Sandbox :public SIByL::Application
{
public:
	Sandbox()
	{
		PushLayer(new ExampleLayer());
	}

	~Sandbox()
	{

	}
};

SIByL::Application* SIByL::CreateApplication()
{
	Renderer::SetRaster(SIByL::RasterRenderer::DirectX12);
	Renderer::SetRayTracer(SIByL::RayTracerRenderer::Cuda);
	SIByL_APP_TRACE("Create Application");
	return new Sandbox();
}