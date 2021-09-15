
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
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Core/Events/KeyEvent.h"
#include "Sibyl/Core/Events/MouseEvent.h"
#include "Sibyl/Core/Events/ApplicationEvent.h"

namespace SIByLEditor
{
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
		m_ActiveScene = CreateRef<Scene>();
		m_SceneHierarchyPanel = CreateRef<SceneHierarchyPanel>(m_ActiveScene);

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

		VertexBufferLayout layout =
		{
			{ShaderDataType::Float3, "POSITION"},
			{ShaderDataType::Float2, "TEXCOORD"},
		};
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

		camera->SetCamera();
		Renderer2D::GetMaterial()->SetPass();

		DrawItemPool& diPool = m_ActiveScene->GetDrawItems();
		for (Ref<DrawItem> drawItem : diPool)
		{
			Graphic::CurrentCamera->OnDrawCall();
			Graphic::CurrentMaterial->OnDrawCall();
			drawItem->OnDrawCall();
		}

		FrameBufferLibrary::Fetch("SceneView")->Unbind();
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
		else if (e.GetKeyCode() == SIByL_KEY_Q)
		{
			if (!Input::IsMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				GizmoType = -1;
		}
		else if (e.GetKeyCode() == SIByL_KEY_W)
		{
			if (!Input::IsMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				GizmoType = 0;
		}
		else if (e.GetKeyCode() == SIByL_KEY_E)
		{
			if (!Input::IsMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				GizmoType = 1;
		}
		else if (e.GetKeyCode() == SIByL_KEY_R)
		{
			if (!Input::IsMouseButtonPressed(SIByL_MOUSE_BUTTON_RIGHT))
				GizmoType = 2;
		}
	}
	void EditorLayer::NewScene()
	{
		m_ActiveScene = CreateRef<Scene>();
		m_FrameBuffer->Resize(m_ViewportSize.x, m_ViewportSize.y);
		camera->Resize(m_ViewportSize.x, m_ViewportSize.y);
		m_SceneHierarchyPanel->SetContext(m_ActiveScene);
	}
	void EditorLayer::OpenScene()
	{
		std::string filepath = FileDialogs::OpenFile("SIByL Scene (*.scene)\0*.scene\0");
		if (!filepath.empty())
		{
			m_ActiveScene = CreateRef<Scene>();
			m_FrameBuffer->Resize(m_ViewportSize.x, m_ViewportSize.y);
			camera->Resize(m_ViewportSize.x, m_ViewportSize.y);
			m_SceneHierarchyPanel->SetContext(m_ActiveScene);

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
		{
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2({ 0,0 }));
			ImGui::Begin("Viewport");

			m_ViewportFocused = ImGui::IsWindowFocused();
			m_ViewportHoverd = ImGui::IsWindowHovered();
			Application::Get().GetImGuiLayer()->SetBlockEvents(!m_ViewportFocused && !m_ViewportHoverd);

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

			Entity selectedEntity = m_SceneHierarchyPanel->GetSelectedEntity();
			if (selectedEntity && GizmoType != -1)
			{
				ImGuizmo::SetOrthographic(false);
				ImGuizmo::SetDrawlist();

				float windowWidth = (float)ImGui::GetWindowWidth();
				float windowHeight = (float)ImGui::GetWindowHeight();
				ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, windowWidth, windowHeight);

				glm::mat4 cameraView = camera->GetViewMatrix();
				glm::mat4 cameraProj = camera->GetProjectionMatrix();

				// Entity transform
				auto& tc = selectedEntity.GetComponent<TransformComponent>();
				glm::mat4 transform = tc.GetTransform();

				// Snapping 
				bool snap = Input::IsKeyPressed(SIByL_KEY_LEFT_CONTROL);
				float snapValue = 0.5f;
				if (GizmoType == ImGuizmo::OPERATION::ROTATE)
					snapValue = 45.0f;

				float snapValues[3] = { snapValue, snapValue ,snapValue };


				ImGuizmo::Manipulate(glm::value_ptr(cameraView), glm::value_ptr(cameraProj),
					ImGuizmo::OPERATION(GizmoType), ImGuizmo::LOCAL, glm::value_ptr(transform), 
					nullptr, snap ? snapValues : nullptr);
				
				if (ImGuizmo::IsUsing())
				{
					glm::vec3 translation, rotation, scale;
					DecomposeTransform(transform, translation, rotation, scale);
					glm::vec3 deltaRotation = rotation - tc.EulerAngles;
					tc.Translation = translation;
					tc.Scale = scale;
					tc.EulerAngles += deltaRotation;
					transform = tc.GetTransform();
				}
			}

			if (ImGui::BeginDragDropTarget())
			{
				const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("SCENE");

				ImGui::EndDragDropTarget();
			}

			ImGui::End();
			ImGui::PopStyleVar();
		}

		//////////////////////////////////////////////
		// Scene Hierarchy
		//////////////////////////////////////////////
		m_SceneHierarchyPanel->OnImGuiRender();
		m_ContentBrowserPanel.OnImGuiRender();

		ImGui::End();
	}
}