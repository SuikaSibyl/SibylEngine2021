#pragma once
#include <SIByL.h>
#include "Editor/Panels/SceneHierarchyPanel.h"
#include "Editor/Panels/ContentBrowserPanel.h"
#include "Editor/Panels/InspectorPanel.h"
#include "Editor/Panels/ViewportPanel.h"
#include "Sibyl/Graphic/Core/Texture/Image.h"

#include "Sibyl/Graphic/AbstractAPI/Library/FrameBufferLibrary.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/FrameConstantsManager.h"

using namespace SIByL;

namespace SIByL
{
	namespace SRenderPipeline
	{
		class SPipeline;
	}
}

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
		Ref<ComputeShader> acesShader;
		Ref<ViewCameraController> viewCameraController;
		Ref<TriangleMesh> triangle;
		Ref<Camera> camera;
		Ref<Camera> orthoCamera;
		Ref<FrameBuffer> m_FrameBuffer;
		Ref<SRenderPipeline::SPipeline> m_ScriptablePipeline;
		Ref<FrameConstantsManager> m_FrameConstants;
		Ref<Scene> m_ActiveScene;
		Ref<Texture2D> texture;
		Ref<Texture2D> texture1;

		static SceneHierarchyPanel	s_SceneHierarchyPanel;
		static ContentBrowserPanel	s_ContentBrowserPanel;
		static InspectorPanel		s_InspectorPanel;
		static ViewportPanel		s_ViewportPanels;

		Entity m_SqureTest;

		/////////////////////
		////   Icons  ////
		/////////////////////
		static Ref<Texture2D> IconFolder;
		static Ref<Texture2D> IconImage;
		static Ref<Texture2D> IconMesh;
		static Ref<Texture2D> IconScene;
		static Ref<Texture2D> IconFile;
		static Ref<Texture2D> IconMaterial;
		static Ref<Texture2D> IconShader;
		static Ref<Texture2D> GetIcon(const std::string& path);

	private:
		bool OnKeyPressed(KeyPressedEvent& e);

		void NewScene();
		void OpenScene();
		void SaveScene();

		enum class RenderType
		{
			Rasterizer,
			RayTracer,
		};
		RenderType mRenderType = RenderType::Rasterizer;
	};

}