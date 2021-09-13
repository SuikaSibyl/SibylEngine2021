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

		~EditorLayer();

		struct VertexData
		{
			float position[3];
			float uv[2];
		};

		void OnInitResource() override;

		void OnDraw() override;

		virtual void OnAttach() override;

		virtual void OnUpdate() override;

		virtual void OnDrawImGui();

		void OnEvent(SIByL::Event& event) override;

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

		static Ref<Texture2D> IconFolder;
		static Ref<Texture2D> IconImage;
		static Ref<Texture2D> IconMesh;
		static Ref<Texture2D> IconScene;
		static Ref<Texture2D> IconFile;

	private:
		bool OnKeyPressed(KeyPressedEvent& e);

		void NewScene();
		void OpenScene();
		void SaveScene();
	};

}