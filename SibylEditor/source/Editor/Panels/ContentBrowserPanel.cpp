#include "SIByLpch.h"
#include "ContentBrowserPanel.h"

#include <imgui.h>
#include <filesystem>

#include "EditorLayer.h"

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

			ImGui::PushID((const void*)path.c_str());
			// If is directory
			Ref<Texture2D> icon = directoryEntry.is_directory() ? SIByLEditor::EditorLayer::IconFolder : SIByLEditor::EditorLayer::IconFile;

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
			ImGui::ImageButton(icon->GetImGuiHandle(), { thumbnailSize,thumbnailSize }, { 0,1 }, { 1,0 });

			if (ImGui::BeginDragDropSource())
			{
				const wchar_t* itemPath = relativePath.c_str();
				ImGui::SetDragDropPayload("CONTENT_BROWSER_ITEM", itemPath, (wcslen(itemPath) + 1) * sizeof(wchar_t), ImGuiCond_Once);
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
			ImGui::TextWrapped(filenameString.c_str());

			ImGui::NextColumn();

			ImGui::PopID();
		}

		ImGui::End();
	}

}