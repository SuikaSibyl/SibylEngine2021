#pragma once
#include <SIByL.h>
#include "Editor/Panels/SceneHierarchyPanel.h"
#include "Editor/Panels/ContentBrowserPanel.h"
#include "Sibyl/Graphic/Core/Texture/Image.h"

#include "Sibyl/Graphic/AbstractAPI/Library/FrameBufferLibrary.h"

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
			//Ref<Image> image = CreateRef<Image>(16, 16, 4, { 1,1,1,1 });
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
			desc.Channel = 4;
			m_FrameBuffer = FrameBuffer::Create(desc, "SceneView");
		}

		void OnDraw() override
		{
			FrameBufferLibrary::Fetch("SceneView")->Bind();
			FrameBufferLibrary::Fetch("SceneView")->ClearBuffer();
			Renderer2D::BeginScene(camera);
			float totalTime = Application::Get().GetFrameTimer()->TotalTime();
			Renderer2D::DrawQuad({ 0,0,0 }, { .2,.2 }, { 0.5 + 0.5 * sin(totalTime),1,1,1 });
			Renderer2D::EndScene();
			m_FrameBuffer->Unbind();
		}

		virtual void OnAttach() override;

		virtual void OnUpdate() override;

		virtual void OnDrawImGui();

		void OnEvent(SIByL::Event& event) override
		{
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
		Ref<SceneHierarchyPanel> m_SceneHierarchyPanel;
		ContentBrowserPanel m_ContentBrowserPanel;

		Entity m_SqureTest;
		/////////////////////
		////   Viewport  ////
		/////////////////////
		glm::vec2 m_ViewportSize;
		bool m_ViewportFocused;
		bool m_ViewportHoverd;
	};

}