#include <iostream>

#include <SIByL.h>

#include <Sibyl/Renderer/Renderer.h>
#include "Sibyl/Renderer/Shader.h"
#include "Sibyl/Graphic/Geometry/TriangleMesh.h"
#include "Sibyl/Graphic/Texture/Texture.h"

//#include "Platform/DirectX12/Common/DX12Context.h"
//#include "Platform/DirectX12/Renderer/DX12ShaderBinder.h"
//#include "Platform/DirectX12/Core/DX12FrameResources.h"
//#include "Platform/DirectX12/Graphic/Texture/DX12Texture.h"

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

	void OnInitResource() override
	{
		VertexBufferLayout layout =
		{
			{ShaderDataType::Float3, "POSITION"},
			{ShaderDataType::Float2, "TEXCOORD"},
		};

		std::vector<ConstantBufferLayout> CBlayouts =
		{
			// ConstantBuffer1
			{
				{ShaderDataType::Float3, "Color"}
			},
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "Main"}
			},
		};

		VertexData vertices[] = {
			0.5f, 0.5f, 0.0f,     1.0f,1.0f,  
			0.5f, -0.5f, 0.0f,    1.0f,0.0f,  
			-0.5f, -0.5f, 0.0f,   0.0f ,0.0f, 
			-0.5f, 0.5f, 0.0f,	  0.0f,1.0f,  
		};

		uint32_t indices[] = { // 注意索引从0开始! 
			0, 1, 3, // 第一个三角形
			1, 2, 3  // 第二个三角形
		};

		shader = Shader::Create("Test/basic", 
			ShaderDesc({ true,layout }), 
			ShaderBinderDesc(CBlayouts, SRlayouts));

		triangle = TriangleMesh::Create((float*)vertices, 4, indices, 6, layout);
		texture = Texture2D::Create("fen4.png");
		//litUni = new DX12FrameResource<LitUniforms>();
	}

	void OnUpdate() override
	{
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
		//shader->Use();

		////D3D12_GPU_VIRTUAL_ADDRESS gpuAddr = litUni->GetCurrentGPUAddress();

		//ID3D12GraphicsCommandList* cmdList = DX12Context::GetDXGraphicCommandList();
		////cmdList->SetGraphicsRootConstantBufferView(0, gpuAddr);

		//DX12Texture2D* dxTexture = dynamic_cast<DX12Texture2D*>(texture.get());
		//DX12ShaderBinder* dxBinder = dynamic_cast<DX12ShaderBinder*>(shader->GetShaderBinder().get());
		//Ref<DynamicDescriptorHeap> dxSDH = dxBinder->GetSrvDynamicDescriptorHeap();
		//dxSDH->StageDescriptors(1, 0, 1, dxTexture->GetSRVHandle());
		//dxSDH->CommitStagedDescriptorsForDraw();

		//texture->Bind(0);
		//triangle->RasterDraw();
	}

	Ref<Shader> shader;
	Ref<TriangleMesh> triangle;
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
	//_CrtSetBreakAlloc(1342);
	Renderer::SetRaster(SIByL::RasterRenderer::DirectX12);
	Renderer::SetRayTracer(SIByL::RayTracerRenderer::Cuda);
	SIByL_APP_TRACE("Create Application");
	return new Sandbox();
}