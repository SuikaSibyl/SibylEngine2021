#include <SIByL.h>

#include <Sibyl/Graphic/Core/Camera.h>
#include <Sibyl/Components/ViewCameraController.h>

using namespace SIByL;

class EditorLayer :public SIByL::Layer
{
public:
	EditorLayer()
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
		texture = Texture2D::Create("fen4.png");
		texture1 = Texture2D::Create("amagami4.png");

		camera = std::make_shared<PerspectiveCamera>(45, 
			Application::Get().GetWindow().GetWidth(), 
			Application::Get().GetWindow().GetHeight());

		orthoCamera = std::make_shared<OrthographicCamera>(
			Application::Get().GetWindow().GetWidth(),
			Application::Get().GetWindow().GetHeight());

		viewCameraController = std::make_shared<ViewCameraController>(camera);

		FrameBufferDesc desc;
		desc.Width = 1280;
		desc.Height = 720;
		m_FrameBuffer = FrameBuffer::Create(desc);
	}

	void OnUpdate() override
	{
		PROFILE_SCOPE_FUNCTION();

		SIByL_CORE_INFO("FPS: {0}, {1} ms", 
			Application::Get().GetFrameTimer()->GetFPS(),
			Application::Get().GetFrameTimer()->GetMsPF());

		viewCameraController->OnUpdate();
	}

	virtual void OnDrawImGui() 
	{
		ImGui::Begin("Setting");
		unsigned int textureID = m_FrameBuffer->GetColorAttachment();
		ImGui::Image((void*)textureID, ImVec2{ 512,512 }, { 1,1 }, { 0,0 });
		ImGui::End();
		static bool show = true;
		//ImGui::ShowDemoWindow(&show);
	}

	void OnEvent(SIByL::Event& event) override
	{
		viewCameraController->OnEvent(event);
	}

	void OnDraw() override
	{
		m_FrameBuffer->Bind();
		Renderer2D::BeginScene(camera);
		Renderer2D::DrawQuad({ 0,0,0 }, { .2,.2 }, texture);
		Renderer2D::EndScene();
		m_FrameBuffer->Unbind();
	}

	Ref<Shader> shader;
	Ref<ViewCameraController> viewCameraController;
	Ref<TriangleMesh> triangle;
	Ref<Texture2D> texture;
	Ref<Texture2D> texture1;
	Ref<Camera> camera;
	Ref<Camera> orthoCamera;
	Ref<FrameBuffer> m_FrameBuffer;
};

class SIByLEditor :public SIByL::Application
{
public:
	SIByLEditor()
	{
		PushLayer(new EditorLayer());
	}

	~SIByLEditor()
	{

	}
};

SIByL::Application* SIByL::CreateApplication()
{
	//_CrtSetBreakAlloc(1330);
	Renderer::SetRaster(SIByL::RasterRenderer::OpenGL);
	Renderer::SetRayTracer(SIByL::RayTracerRenderer::Cuda);
	SIByL_APP_TRACE("Create Application");
	return new SIByLEditor();
}