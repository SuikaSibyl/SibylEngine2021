#include <SIByL.h>

#include <Sibyl/Graphic/Core/Camera.h>

using namespace SIByL;

class ExampleLayer :public SIByL::Layer
{
public:
	ExampleLayer()
		:Layer("Example")
	{

	}

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
				{ShaderDataType::Mat4, "Model"},
				{ShaderDataType::Mat4, "View"},
				{ShaderDataType::Mat4, "Projection"},
				{ShaderDataType::Float3, "Color"},
			},
		};

		std::vector<ShaderResourceLayout> SRlayouts =
		{
			// ShaderResourceTable 1
			{
				{ShaderResourceType::Texture2D, "Main"},
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
		texture1 = Texture2D::Create("amagami4.png");

		camera = std::make_shared<PerspectiveCamera>(45, 
			Application::Get().GetWindow().GetWidth(), 
			Application::Get().GetWindow().GetHeight());
	}

	void OnUpdate() override
	{
		if (SIByL::Input::IsKeyPressed(SIByL_KEY_A))
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
		glm::mat4 model = glm::mat4(1.0f);
		//model = glm::scale(model, { 2,2,2 });

		float x = Input::GetMouseX();
		std::cout << x << std::endl;

		float totalTime = Application::Get().GetFrameTimer()->TotalTime();
		shader->Use();
		shader->GetBinder()->SetFloat3("Color", { sin(totalTime),1,0 });
		shader->GetBinder()->SetMatrix4x4("Model", model);
		shader->GetBinder()->SetMatrix4x4("View", glm::transpose(camera->GetViewMatrix()));
		shader->GetBinder()->SetMatrix4x4("Projection", (camera->GetProjectionMatrix()));
		
		shader->GetBinder()->TEMPUpdateAllConstants();
		if (((int)(totalTime) % 2) == 0)
			shader->GetBinder()->SetTexture2D("Main", texture);
		else
			shader->GetBinder()->SetTexture2D("Main", texture1);
		shader->GetBinder()->TEMPUpdateAllResources();

		triangle->RasterDraw();
	}

	Ref<Shader> shader;
	Ref<TriangleMesh> triangle;
	Ref<Texture2D> texture;
	Ref<Texture2D> texture1;
	Ref<Camera> camera;
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
	//_CrtSetBreakAlloc(1330);
	Renderer::SetRaster(SIByL::RasterRenderer::DirectX12);
	Renderer::SetRayTracer(SIByL::RayTracerRenderer::Cuda);
	SIByL_APP_TRACE("Create Application");
	return new Sandbox();
}