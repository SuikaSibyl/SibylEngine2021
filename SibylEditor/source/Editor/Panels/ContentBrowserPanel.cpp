#include "SIByLpch.h"
#include "ContentBrowserPanel.h"

#include <imgui.h>

#include "EditorLayer.h"

#include "Sibyl/ECS/Asset/AssetUtility.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Material.h"

namespace SIByL
{
	constexpr const char* s_AssetsDirectory = "../Assets";

	ContentBrowserPanel::ContentBrowserPanel()
		:m_CurrentDirectory(s_AssetsDirectory)
	{

	}

	void ContentBrowserPanel::OnImGuiRender()
	{
		ImGui::Begin("Content Browser");

		static std::filesystem::path root(s_AssetsDirectory);
		if (root.compare(m_CurrentDirectory))
		{
			if (ImGui::Button("<-"))
			{
				m_CurrentDirectory = m_CurrentDirectory.parent_path();
			}
		}

		static float dpi = Application::Get().GetWindow().GetHighDPI();
		static float padding = 16.0f * dpi;
		static float thumbnailSize = 64.f * dpi;
		float cellSize = thumbnailSize + padding;

		float panelWidth = ImGui::GetContentRegionAvail().x;
		int columnCount = (int)(panelWidth / cellSize);
		if (columnCount < 1)
			columnCount = 1;

		ImGui::Columns(columnCount, 0, false);


		for (auto& directoryEntry : std::filesystem::directory_iterator(m_CurrentDirectory))
		{
			const auto& path = directoryEntry.path();
			std::string pathString = path.string();
			auto relativePath = std::filesystem::relative(directoryEntry.path(), root);
			std::string relativePathString = relativePath.string();
			std::string filenameString = relativePath.filename().string();

			// Omit cache file folder
			if (filenameString == "Cache") continue;

			ImGui::PushID(filenameString.c_str());
			// If is directory
			Ref<Texture2D> icon = directoryEntry.is_directory() ? SIByLEditor::EditorLayer::IconFolder : SIByLEditor::EditorLayer::GetIcon(filenameString);

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
			ImGui::ImageButton(icon->GetImGuiHandle(), { thumbnailSize,thumbnailSize }, { 0,1 }, { 1,0 });

			if (ImGui::BeginDragDropSource())
			{
				ImGui::Text("Dragged");

				const wchar_t* itemPath = relativePath.c_str();
				ImGui::SetDragDropPayload("ASSET", itemPath, (wcslen(itemPath) + 1) * sizeof(wchar_t));
				ImGui::EndDragDropSource();
			}

			ImGui::PopStyleColor();

			if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
			{
				if (directoryEntry.is_directory())
				{
					m_CurrentDirectory /= path.filename();
				}
			}

			if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Left))
			{
				AssetType type = GetAssetType(filenameString);
				switch (type)
				{
				case SIByL::AssetType::Unknown:
					break;
				case SIByL::AssetType::Material:
				{
					Ref<Material> mat = GetAssetByPath<Material>(relativePathString);
					SIByLEditor::EditorLayer::s_InspectorPanel.SetSelectedMaterial(mat);
					break;
				}
				default:
					break;
				}
			}

			ImGui::TextWrapped(filenameString.c_str());

			ImGui::NextColumn();

			ImGui::PopID();
		}

		////////////////////////////////////////////////////////////////////////////
		//							   Add Material Tab						      //
		////////////////////////////////////////////////////////////////////////////
		static bool NewMaterial = false;
		// Right-click on blank space
		if (ImGui::BeginPopupContextWindow(0, 1, false))
		{
			if (ImGui::MenuItem("Create New Material"))
			{
				NewMaterial = true;
			}

			ImGui::EndPopup();
		}

		if (NewMaterial)
		{
			Ref<Texture2D> icon = SIByLEditor::EditorLayer::IconMaterial;

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
			ImGui::ImageButton(icon->GetImGuiHandle(), { thumbnailSize,thumbnailSize }, { 0,1 }, { 1,0 });
			ImGui::PopStyleColor();

			//ImGui::ShowDemoWindow
			static char buf[32] = "DefaultMaterial";
			//static char buf[32] = u8"NIHONGO"; // <- this is how you would write it with C++11, using real kanjis
			if (ImGui::InputText(" ", buf, IM_ARRAYSIZE(buf), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll))
			{
				NewMaterial = false;
				Ref<Material> newMat = CreateRef<Material>();
				MaterialSerializer matSerializer(newMat);
				std::string fullPath = m_CurrentDirectory.string() + "/" + std::string(buf) + ".mat";
				matSerializer.Serialize(fullPath);
			}

			//ImGui::TextWrapped(filenameString.c_str());
		}

		ImGui::End();
	}

}