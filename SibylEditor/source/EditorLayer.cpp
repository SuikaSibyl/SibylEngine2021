
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "EditorLayer.h"
#include "Sibyl/ImGui/ImGuiUtility.h"

#include "Sibyl/ECS/Components/Render/SpriteRenderer.h"
#include "Sibyl/Graphic/Core/Texture/Image.h"
#include "Sibyl/ECS/Scene/SceneSerializer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"

namespace SIByLEditor
{
	Ref<Texture2D> EditorLayer::IconFolder = nullptr;
	Ref<Texture2D> EditorLayer::IconImage  = nullptr;
	Ref<Texture2D> EditorLayer::IconMesh   = nullptr;
	Ref<Texture2D> EditorLayer::IconScene  = nullptr;
	Ref<Texture2D> EditorLayer::IconFile   = nullptr;

	Ref<TriangleMesh> m_Mesh = nullptr;

	void EditorLayer::OnAttach()
	{
		m_ActiveScene = CreateRef<Scene>();
		m_SceneHierarchyPanel = CreateRef<SceneHierarchyPanel>(m_ActiveScene);

		Entity square = m_ActiveScene->CreateEntity();
		SpriteRendererComponent& spriteRenderer = square.AddComponent<SpriteRendererComponent>();
		m_SqureTest = square;
		Entity hello = m_ActiveScene->CreateEntity("Hello");

		SceneSerializer serialzier(m_ActiveScene);
		serialzier.Serialize("../Assets/Scenes/Example.scene");

		Image image(8, 8, 4, { 0.1,0.2,0.3,1 });

		IconFolder = Texture2D::Create("../SibylEditor/assets/icons/folder.png");
		IconImage = Texture2D::Create("../SibylEditor/assets/icons/image.png");
		IconMesh = Texture2D::Create("../SibylEditor/assets/icons/mesh.png");
		IconScene = Texture2D::Create("../SibylEditor/assets/icons/scene.png");
		IconFile = Texture2D::Create("../SibylEditor/assets/icons/file.png");

		IconFolder->RegisterImGui();
		IconImage->RegisterImGui();
		IconMesh->RegisterImGui();
		IconScene->RegisterImGui();
		IconFile->RegisterImGui();
	}

	EditorLayer::~EditorLayer()
	{
		IconFolder = nullptr;
		IconImage = nullptr;
		IconMesh = nullptr;
	}

	void EditorLayer::OnInitResource()
	{
		texture = Texture2D::Create(TexturePath + "fen4.png");
		texture1 = Texture2D::Create(TexturePath + "amagami4.png");

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

		SpriteRendererComponent& spriteRenderer = m_SqureTest.GetComponent<SpriteRendererComponent>();
		spriteRenderer.Material = Renderer2D::GetMaterial();

		VertexBufferLayout layout =
		{
			{ShaderDataType::Float3, "POSITION"},
			{ShaderDataType::Float2, "TEXCOORD"},
		};
		//if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
		//{
		//	SIByL_CORE_ERROR("Mesh Load Error");
		//	return;
		//}
	}

	void EditorLayer::OnUpdate()
	{
		PROFILE_SCOPE_FUNCTION();

		m_ActiveScene->OnUpdate();

		//SIByL_CORE_INFO("FPS: {0}, {1} ms",
		//	Application::Get().GetFrameTimer()->GetFPS(),
		//	Application::Get().GetFrameTimer()->GetMsPF());

		if (m_ViewportFocused)
			viewCameraController->OnUpdate();
	}

	void EditorLayer::OnDraw()
	{
		FrameBufferLibrary::Fetch("SceneView")->Bind();
		FrameBufferLibrary::Fetch("SceneView")->ClearBuffer();
		Renderer2D::BeginScene(camera);
		float totalTime = Application::Get().GetFrameTimer()->TotalTime();
		TransformComponent& trans = m_SqureTest.GetComponent<TransformComponent>();
		Renderer2D::DrawQuad(trans.GetTransform(), Renderer2D::GetMaterial());
		Renderer2D::EndScene();
		m_FrameBuffer->Unbind();
	}

	void EditorLayer::OnDrawImGui()
	{
		static bool dockspaceOpen = true;
		static bool opt_fullscreen = true;
		static bool opt_padding = false;
		static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_None;

		// We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
		// because it would be confusing to have two docking targets within each others.
		ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
		if (opt_fullscreen)
		{
			const ImGuiViewport* viewport = ImGui::GetMainViewport();
			ImGui::SetNextWindowPos(viewport->WorkPos);
			ImGui::SetNextWindowSize(viewport->WorkSize);
			ImGui::SetNextWindowViewport(viewport->ID);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
			ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
			window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
			window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
		}
		else
		{
			dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
		}

		// When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
		// and handle the pass-thru hole, so we ask Begin() to not render a background.
		if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
			window_flags |= ImGuiWindowFlags_NoBackground;

		// Important: note that we proceed even if Begin() returns false (aka window is collapsed).
		// This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
		// all active windows docked into it will lose their parent and become undocked.
		// We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
		// any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
		if (!opt_padding)
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
		ImGui::Begin("DockSpace Demo", &dockspaceOpen, window_flags);
		if (!opt_padding)
			ImGui::PopStyleVar();

		if (opt_fullscreen)
			ImGui::PopStyleVar(2);

		// Submit the DockSpace
		ImGuiIO& io = ImGui::GetIO();
		ImGuiStyle& style = ImGui::GetStyle();
		//style.WindowMinSize.x = 350.0f;
		if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
		{
			ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
			ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
		}
		//style.WindowMinSize.x = 32.0f;

		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("Options"))
			{
				// Disabling fullscreen would allow the window to be moved to the front of other windows,
				// which we can't undo at the moment without finer window depth/z control.
				ImGui::MenuItem("Fullscreen", NULL, &opt_fullscreen);
				ImGui::MenuItem("Padding", NULL, &opt_padding);
				ImGui::Separator();

				if (ImGui::MenuItem("Flag: NoSplit", "", (dockspace_flags & ImGuiDockNodeFlags_NoSplit) != 0)) { dockspace_flags ^= ImGuiDockNodeFlags_NoSplit; }
				if (ImGui::MenuItem("Flag: NoResize", "", (dockspace_flags & ImGuiDockNodeFlags_NoResize) != 0)) { dockspace_flags ^= ImGuiDockNodeFlags_NoResize; }
				if (ImGui::MenuItem("Flag: NoDockingInCentralNode", "", (dockspace_flags & ImGuiDockNodeFlags_NoDockingInCentralNode) != 0)) { dockspace_flags ^= ImGuiDockNodeFlags_NoDockingInCentralNode; }
				if (ImGui::MenuItem("Flag: AutoHideTabBar", "", (dockspace_flags & ImGuiDockNodeFlags_AutoHideTabBar) != 0)) { dockspace_flags ^= ImGuiDockNodeFlags_AutoHideTabBar; }
				if (ImGui::MenuItem("Flag: PassthruCentralNode", "", (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode) != 0, opt_fullscreen)) { dockspace_flags ^= ImGuiDockNodeFlags_PassthruCentralNode; }
				ImGui::Separator();

				if (ImGui::MenuItem("Close", NULL, false, &dockspaceOpen != NULL))
					dockspaceOpen = false;
				ImGui::EndMenu();
			}

			ImGui::EndMenuBar();
		}

		bool showdemo = true;
		ImGui::ShowDemoWindow(&showdemo);
		//////////////////////////////////////////////
		// View Ports
		//////////////////////////////////////////////
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2({ 0,0 }));
			ImGui::Begin("Viewport");

			m_ViewportFocused = ImGui::IsWindowFocused();
			m_ViewportHoverd = ImGui::IsWindowHovered();
			Application::Get().GetImGuiLayer()->SetBlockEvents(!m_ViewportFocused || !m_ViewportHoverd);

			ImVec2 viewportPanelSize = ImGui::GetContentRegionAvail();
			if (m_ViewportSize != *((glm::vec2*)&viewportPanelSize))
			{
				// Viewport Change Size
				m_ViewportSize = { viewportPanelSize.x, viewportPanelSize.y };
				m_FrameBuffer->Resize(viewportPanelSize.x, viewportPanelSize.y);
				camera->Resize(viewportPanelSize.x, viewportPanelSize.y);
			}

			//unsigned int textureID = m_FrameBuffer->GetColorAttachment();

			ImGui::DrawImage((void*)m_FrameBuffer->GetColorAttachment(), ImVec2{
				viewportPanelSize.x,
				viewportPanelSize.y });

			if (ImGui::BeginDragDropTarget())
			{
				const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE");

				ImGui::EndDragDropTarget();
			}

			ImGui::End();
			ImGui::PopStyleVar();
		}

		ImGui::Begin("Debug");
		if (ImGui::Button("What"))
		{
			Ref<Material> mat = Renderer2D::GetMaterial();
			mat->SetFloat4("Color", { 0,1,0,1 });
			mat->SetTexture2D("Main", Texture2D::Create("Resources/Textures/checkboard.png"));
		}
		ImGui::End();

		//////////////////////////////////////////////
		// Scene Hierarchy
		//////////////////////////////////////////////
		m_SceneHierarchyPanel->OnImGuiRender();
		m_ContentBrowserPanel.OnImGuiRender();

		ImGui::End();
	}
}