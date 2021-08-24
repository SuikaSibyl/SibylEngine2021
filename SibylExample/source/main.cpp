#include <iostream>

#include <SIByL.h>

#include <Sibyl/Renderer/Renderer.h>
#include "Sibyl/Renderer/Shader.h"
#include "Sibyl/Graphic/Geometry/TriangleMesh.h"

#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/Renderer/DX12ShaderBinder.h"
#include "Platform/DirectX12/Core/DX12FrameResources.h"
#include "Sibyl/Graphic/Texture/Texture.h"
using namespace SIByL;

class ExampleLayer :public SIByL::Layer
{
public:
	ExampleLayer()
		:Layer("Example")
	{

	}

	struct LitUniforms
	{
		float color[3];
	};


	struct VertexData
	{
		float position[3];
		float uv[2];
	};

	void OnInitRenderer() override
	{
		VertexBufferLayout layout =
		{
			{ShaderDataType::Float3, "POSITION"},
			{ShaderDataType::Float2, "UV"},
		};

		VertexData vertices[] = {
			0.5f, 0.5f, 0.0f,     1,1,
			0.5f, -0.5f, 0.0f,    1,0,
			-0.5f, -0.5f, 0.0f,   0,0,
			-0.5f, 0.5f, 0.0f,	  0,1,
		};

		uint32_t indices[] = { // 注意索引从0开始! 
			0, 1, 3, // 第一个三角形
			1, 2, 3  // 第二个三角形
		};

		shader = Shader::Create("Test/basic", "Test/basic");
		shader->CreateBinder(layout);
		triangle = TriangleMesh::Create((float*)vertices, 4, indices, 6, layout);
		texture = Texture2D::Create("TEST.png");
	}

	void OnUpdate() override
	{
		//SIByL_APP_INFO("ExampleLayer::Update");
		if (SIByL::Input::IsKeyPressed(SIByL_KEY_TAB))
			SIByL_APP_TRACE("Tab key is pressed!");

		SIByL_CORE_INFO("FPS: {0}, {1} ms", 
			Application::Get().GetFrameTimer()->GetFPS(),
			Application::Get().GetFrameTimer()->GetMsPF());

		//Ref<LitUniforms> lu = litUni->GetCurrentBuffer();
		//lu->color[0] = sin(Application::Get().GetFrameTimer()->TotalTime());
		//lu->color[1] = 0.5;
		//lu->color[2] = 0.8;
		//litUni->UploadCurrentBuffer();
	}

	void OnEvent(SIByL::Event& event) override
	{
		//SIByL_APP_TRACE("{0}", event);
	}

	void OnDraw() override
	{
		shader->Use();

		//D3D12_GPU_VIRTUAL_ADDRESS gpuAddr = litUni->GetCurrentGPUAddress();

		//ID3D12GraphicsCommandList* cmdList = DX12Context::GetDXGraphicCommandList();
		//cmdList->SetGraphicsRootConstantBufferView(0, gpuAddr);
		texture->Bind(0);
		triangle->RasterDraw();
	}

	Shader* shader;
	TriangleMesh* triangle;
	Ref<Texture2D> texture;
	//DX12FrameResource<LitUniforms>* litUni;
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
	Renderer::SetRaster(SIByL::RasterRenderer::OpenGL);
	Renderer::SetRayTracer(SIByL::RayTracerRenderer::Cuda);
	SIByL_APP_TRACE("Create Application");
	return new Sandbox();
}