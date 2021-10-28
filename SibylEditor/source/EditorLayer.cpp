#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include "EditorLayer.h"
#include "Sibyl/ImGui/ImGuiUtility.h"
#include "Sibyl/Basic/Utils/PlatformUtils.h"
#include "Sibyl/Basic/Utils/MathUtils.h"
#include "Sibyl/ECS/Components/Render/SpriteRenderer.h"
#include "Sibyl/Graphic/Core/Texture/Image.h"
#include "Sibyl/ECS/Scene/SceneSerializer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/DrawItem.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/ComputeInstance.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/Graphic/Core/Lighting/LightManager.h"
#include "Sibyl/Core/Events/KeyEvent.h"
#include "Sibyl/Core/Events/MouseEvent.h"
#include "Sibyl/Core/Events/ApplicationEvent.h"

#ifdef SIBYL_PLATFORM_CUDA
#include <CudaModule/source/CudaModule.h>
#endif // SIBYL_PLATFORM_CUDA
#include <NetworkModule/include/NetworkModule.h>

#include "Sibyl/ECS/UniqueID/UniqueID.h"

namespace SIByLEditor
{
	SceneHierarchyPanel EditorLayer::s_SceneHierarchyPanel;
	ContentBrowserPanel EditorLayer::s_ContentBrowserPanel;
	InspectorPanel		EditorLayer::s_InspectorPanel;
	ViewportPanel		EditorLayer::s_ViewportPanels;

	Ref<Texture2D> EditorLayer::IconFolder = nullptr;
	Ref<Texture2D> EditorLayer::IconImage  = nullptr;
	Ref<Texture2D> EditorLayer::IconMesh   = nullptr;
	Ref<Texture2D> EditorLayer::IconScene  = nullptr;
	Ref<Texture2D> EditorLayer::IconFile   = nullptr;
	Ref<Texture2D> EditorLayer::IconMaterial = nullptr;
	Ref<Texture2D> EditorLayer::IconShader = nullptr;

	Ref<TriangleMesh> m_Mesh = nullptr;

	Ref<Texture2D> EditorLayer::GetIcon(const std::string& path)
	{
		std::string::size_type pos = path.rfind('.');
		std::string ext = path.substr(pos == std::string::npos ? path.length() : pos + 1);
		if (ext == "mat")
			return EditorLayer::IconMaterial;
		else if (ext == "fbx" || ext == "FBX")
			return EditorLayer::IconMesh;
		else if (ext == "png" || ext == "jpg")
			return EditorLayer::IconImage;
		else if (ext == "scene")
			return EditorLayer::IconScene;
		else if (ext == "glsl" || ext == "hlsl")
			return EditorLayer::IconShader;

		return EditorLayer::IconFile;
	}

	void EditorLayer::OnAttach()
	{
		uint64_t uuid = UniqueID::RequestUniqueID();
		m_ActiveScene = CreateRef<Scene>();
		s_SceneHierarchyPanel = SceneHierarchyPanel(m_ActiveScene);

		Image image(8, 8, 4, { 0.1,0.2,0.3,1 });

		IconFolder = Texture2D::Create("../SibylEditor/assets/icons/folder.png");
		IconImage = Texture2D::Create("../SibylEditor/assets/icons/image.png");
		IconMesh = Texture2D::Create("../SibylEditor/assets/icons/mesh.png");
		IconScene = Texture2D::Create("../SibylEditor/assets/icons/scene.png");
		IconFile = Texture2D::Create("../SibylEditor/assets/icons/file.png");
		IconMaterial = Texture2D::Create("../SibylEditor/assets/icons/material.png");
		IconShader = Texture2D::Create("../SibylEditor/assets/icons/shader.png");

		IconFolder->RegisterImGui();
		IconImage->RegisterImGui();
		IconMesh->RegisterImGui();
		IconScene->RegisterImGui();
		IconFile->RegisterImGui();
		IconMaterial->RegisterImGui();
		IconShader->RegisterImGui();

		m_FrameConstants = CreateRef<FrameConstantsManager>();
		m_FrameConstants->SetFrame();
		LightManager::SetFrameConstantsManager(m_FrameConstants.get());
	}

	EditorLayer::~EditorLayer()
	{
		IconFolder = nullptr;
		IconImage = nullptr;
		IconMesh = nullptr;
		IconScene = nullptr;
		IconFile = nullptr;
		IconMaterial = nullptr;
		IconShader = nullptr;

		s_SceneHierarchyPanel = SceneHierarchyPanel();
		s_ContentBrowserPanel = ContentBrowserPanel();
		s_InspectorPanel = InspectorPanel();
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
		desc.Formats = { 
			FrameBufferTextureFormat::RGB8, 
			FrameBufferTextureFormat::R16G16F,
			FrameBufferTextureFormat::DEPTH24STENCIL8 };
		// Frame Buffer 0: Main Render Buffer
		m_FrameBuffer = FrameBuffer::Create(desc, "SceneView");
		
		desc.Formats = {FrameBufferTextureFormat::RGB8};

		s_ViewportPanels.SetCamera(camera);
		s_ViewportPanels.SetFrameBuffer(m_FrameBuffer);

		// ===================================================================
		// Frame Buffer 1: ACES
		Ref<FrameBuffer> m_PostProcessBuffer_1 = FrameBuffer::Create(desc, "POST1");
		Ref<ComputeInstance> ACESInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\ACES"));
		Library<ComputeInstance>::Push("ACES", ACESInstance);
		ACESInstance->SetRenderTarget2D("ACESResult", m_PostProcessBuffer_1, 0);
		ACESInstance->SetTexture2D("Input", m_FrameBuffer->GetRenderTarget(0));
		ACESInstance->SetFloat("Para", 0.5);

		// ===================================================================
		// Frame Buffer 2/3: TAA
		m_FrameBuffer_TAA[0] = FrameBuffer::Create(desc, "TAA1");
		m_FrameBuffer_TAA[1] = FrameBuffer::Create(desc, "TAA2");
		Ref<ComputeInstance> TAAInstance = CreateRef<ComputeInstance>(Library<ComputeShader>::Fetch("FILE=Shaders\\Compute\\TAA"));
		Library<ComputeInstance>::Push("TAA", TAAInstance);
		TAAInstance->SetRenderTarget2D("TAAResult", m_FrameBuffer_TAA[1], 0);
		TAAInstance->SetTexture2D("u_PreviousFrame", m_FrameBuffer_TAA[0]->GetRenderTarget(0));
		TAAInstance->SetTexture2D("u_CurrentFrame", m_PostProcessBuffer_1->GetRenderTarget(0));
		TAAInstance->SetTexture2D("u_Offset", m_FrameBuffer->GetRenderTarget(1));
		TAAInstance->SetFloat("Alpha", 0.5);
	}

	void EditorLayer::OnUpdate()
	{
		PROFILE_SCOPE_FUNCTION();

		m_ActiveScene->OnUpdate();

		//SIByL_CORE_INFO("FPS: {0}, {1} ms",
		//	Application::Get().GetFrameTimer()->GetFPS(),
		//	Application::Get().GetFrameTimer()->GetMsPF());

		if (s_ViewportPanels.IsViewportFocusd())
			viewCameraController->OnUpdate();
	}

	static bool NextFrame = false;
	void EditorLayer::OnDraw()
	{
		Ref<FrameBuffer> viewportBuffer = Library<FrameBuffer>::Fetch("SceneView");
		viewportBuffer->Bind();
		viewportBuffer->ClearBuffer();

		camera->SetCamera();
		camera->RecordVPMatrix();
		static unsigned int HaltonIndex = 0;
		auto [x, y] = Halton::Halton23(HaltonIndex++);
		camera->Dither(x, y);

		DrawItemPool& diPool = m_ActiveScene->GetDrawItems();
		for (Ref<DrawItem> drawItem : diPool)
		{
			drawItem->m_Material->SetPass();
			Graphic::CurrentCamera->OnDrawCall();
			Graphic::CurrentMaterial->OnDrawCall();
			Graphic::CurrentFrameConstantsManager->OnDrawCall();
			drawItem->OnDrawCall();
		}

		viewportBuffer->Unbind();

		Ref<ComputeInstance> ACESInstance = Library<ComputeInstance>::Fetch("ACES");
		ACESInstance->Dispatch(s_ViewportPanels.GetViewportSize().x, s_ViewportPanels.GetViewportSize().y, 1);

		static int taaBufferIdx = 0;
		Ref<ComputeInstance> TAAInstance = Library<ComputeInstance>::Fetch("TAA");
		TAAInstance->SetRenderTarget2D("TAAResult", m_FrameBuffer_TAA[taaBufferIdx], 0);
		TAAInstance->SetTexture2D("u_PreviousFrame", m_FrameBuffer_TAA[(taaBufferIdx + 1) % 2]->GetRenderTarget(0));
		TAAInstance->SetTexture2D("u_CurrentFrame", FrameBufferLibrary::GetRenderTarget("POST10"));
		taaBufferIdx++; if (taaBufferIdx == 2) taaBufferIdx = 0;
		TAAInstance->Dispatch(s_ViewportPanels.GetViewportSize().x, s_ViewportPanels.GetViewportSize().y, 1);
	}

	void EditorLayer::OnEvent(SIByL::Event& event)
	{
		EventDispatcher dispatcher(event);
		dispatcher.Dispatch<KeyPressedEvent>(BIND_EVENT_FN(EditorLayer::OnKeyPressed));
		
	}

	bool EditorLayer::OnKeyPressed(KeyPressedEvent& e)
	{
		// Shortcuts
		if (e.GetRepeatCount() > 0)
			return false;


		bool control = Input::IsKeyPressed(SIByL_KEY_LEFT_CONTROL) || Input::IsKeyPressed(SIByL_KEY_RIGHT_CONTROL);
		bool shift = Input::IsKeyPressed(SIByL_KEY_LEFT_SHIFT) || Input::IsKeyPressed(SIByL_KEY_RIGHT_SHIFT);

		if (e.GetKeyCode() == SIByL_KEY_N)
		{
			if (control)
			{
				NewScene();
			}
		}
		else if (e.GetKeyCode() == SIByL_KEY_O)
		{
			if (control)
			{
				OpenScene();
			}
		}
		else if (e.GetKeyCode() == SIByL_KEY_S)
		{
			if (control && shift)
			{
				SaveScene();
			}
		}
		else if (e.GetKeyCode() == SIByL_KEY_T)
		{
			NextFrame = true;
		}
		s_ViewportPanels.OnKeyPressed(e);
	}

	void EditorLayer::NewScene()
	{
		m_ActiveScene = CreateRef<Scene>();
		FrameBufferLibrary::ResizeAll(s_ViewportPanels.GetViewportSize().x, s_ViewportPanels.GetViewportSize().y);
		camera->Resize(s_ViewportPanels.GetViewportSize().x, s_ViewportPanels.GetViewportSize().y);
		s_SceneHierarchyPanel.SetContext(m_ActiveScene);
	}

	void EditorLayer::OpenScene()
	{
		std::string filepath = FileDialogs::OpenFile("SIByL Scene (*.scene)\0*.scene\0");
		if (!filepath.empty())
		{
			m_ActiveScene = CreateRef<Scene>();
			FrameBufferLibrary::ResizeAll(s_ViewportPanels.GetViewportSize().x, s_ViewportPanels.GetViewportSize().y);
			camera->Resize(s_ViewportPanels.GetViewportSize().x, s_ViewportPanels.GetViewportSize().y);
			s_SceneHierarchyPanel.SetContext(m_ActiveScene);

			SceneSerializer serialzier(m_ActiveScene);
			serialzier.Deserialize(filepath);
		}
	}
	void EditorLayer::SaveScene()
	{
		std::string filepath = FileDialogs::SaveFile("SIByL Scene (*.scene)\0*.scene\0");
		if (!filepath.empty())
		{
			SceneSerializer serialzier(m_ActiveScene);
			serialzier.Serialize(filepath);
		}
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

		//////////////////////////////////////////////
		// MenuBar
		//////////////////////////////////////////////
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("New", "Ctrl+N"))
				{
					NewScene();
				}

				if (ImGui::MenuItem("Save Scene", "Ctrl+Shift+S"))
				{
					SaveScene();
				}

				if (ImGui::MenuItem("Load Scene", "Ctrl+O"))
				{
					OpenScene();
				}

				ImGui::EndMenu();
			}

			ImGui::EndMenuBar();
		}

		bool showdemo = true;
		ImGui::ShowDemoWindow(&showdemo);

		//////////////////////////////////////////////
		// View Ports
		//////////////////////////////////////////////
		s_ViewportPanels.OnImGuiRender();

		//////////////////////////////////////////////
		// Scene Hierarchy
		//////////////////////////////////////////////
		s_SceneHierarchyPanel.OnImGuiRender();
		s_ContentBrowserPanel.OnImGuiRender();
		s_InspectorPanel.OnImGuiRender();

		ImGui::End();

		ImGui::Begin("Post Processing");
		static float ACESPara = 0.5;
		if (ImGui::DragFloat("ACES Para", &ACESPara, 0.1, 0, 10))
		{
			std::cout << "Para: " << ACESPara << std::endl;
			Ref<ComputeInstance> ACESInstance = Library<ComputeInstance>::Fetch("ACES");
			ACESInstance->SetFloat("Para", ACESPara);
		}
		ImGui::End();
	}
}