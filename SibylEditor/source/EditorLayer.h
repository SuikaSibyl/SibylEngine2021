#pragma once
#include <SIByL.h>

using namespace SIByL;

namespace SIByLEditor
{

	class EditorLayer :public SIByL::Layer
	{
	public:
		EditorLayer()
			:Layer("Editor")
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

		virtual void OnAttach() override;

		virtual void OnUpdate() override;

		virtual void OnDrawImGui();

		void OnEvent(SIByL::Event& event) override
		{
			//viewCameraController->OnEvent(event);
		}

		void OnDraw() override
		{
			m_FrameBuffer->Bind();
			m_FrameBuffer->ClearBuffer();
			Renderer2D::BeginScene(camera);
			Renderer2D::DrawQuad({ 0,0,0 }, { .2,.2 }, texture);
			Renderer2D::EndScene();
			m_FrameBuffer->Unbind();


			Renderer2D::BeginScene(camera);
			Renderer2D::DrawQuad({ 0,0,0 }, { .2,.2 }, texture);
			Renderer2D::EndScene();
		}

		Ref<Shader> shader;
		Ref<ViewCameraController> viewCameraController;
		Ref<TriangleMesh> triangle;
		Ref<Texture2D> texture;
		Ref<Texture2D> texture1;
		Ref<Camera> camera;
		Ref<Camera> orthoCamera;
		Ref<FrameBuffer> m_FrameBuffer;
		Ref<Scene> m_ActiveScene;

		/////////////////////
		////   Viewport  ////
		/////////////////////
		glm::vec2 m_ViewportSize;
		bool m_ViewportFocused;
		bool m_ViewportHoverd;
	};

}