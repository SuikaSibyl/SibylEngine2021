module;
#include <vector>
#include <string>
#include <filesystem>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <functional>
#include "entt.hpp"
export module Editor.ContentBrowser;
import Core.Window;
import Editor.Widget;
import GFX.Texture;
import Asset.AssetLayer;
import Editor.ImGuiLayer;
import Editor.ImImage;

namespace SIByL::Editor
{
	export struct ContentBrowser :public Widget
	{
		ContentBrowser(WindowLayer* window_layer, Asset::AssetLayer* asset_layer, ImGuiLayer* imgui_layer);
		virtual auto onDrawGui() noexcept -> void override;
		auto getIcon(const std::string& path) noexcept -> ImImage*;

		float dpi = 1;
		std::filesystem::path currentDirectory;

		ImImage* folderIm;
		ImImage* meshIm;
		ImImage* sceneIm;
		ImImage* fileIm;
		ImImage* materialIm;
		ImImage* shaderIm;
		ImImage* imageIm;
		GFX::Texture iconFolder;
		GFX::Texture iconMesh;
		GFX::Texture iconScene;
		GFX::Texture iconFile;
		GFX::Texture iconMaterial;
		GFX::Texture iconShader;
		GFX::Texture iconImage;
	};

	ContentBrowser::ContentBrowser(WindowLayer* window_layer, Asset::AssetLayer* asset_layer, ImGuiLayer* imgui_layer)
		:currentDirectory("./assets")
	{
		dpi = window_layer->getWindow()->getHighDPI();
		iconFolder	 = GFX::Texture::query(13249315409853959290, asset_layer);
		iconMesh	 = GFX::Texture::query(11971981965541002362, asset_layer);
		iconScene	 = GFX::Texture::query(14779131908276374650, asset_layer);
		iconFile	 = GFX::Texture::query(2267850668464426106, asset_layer);
		iconMaterial = GFX::Texture::query(660628551446580346, asset_layer);
		iconShader   = GFX::Texture::query(8651421665285393530, asset_layer);
		iconImage    = GFX::Texture::query(12673417607503957114, asset_layer);

		folderIm = imgui_layer->getImImage(iconFolder);
		meshIm = imgui_layer->getImImage(iconMesh);
		sceneIm = imgui_layer->getImImage(iconScene);
		fileIm = imgui_layer->getImImage(iconFile);
		materialIm = imgui_layer->getImImage(iconMaterial);
		shaderIm = imgui_layer->getImImage(iconShader);
		imageIm = imgui_layer->getImImage(iconImage);
	}

	auto ContentBrowser::getIcon(std::string const& path) noexcept -> ImImage*
	{
		std::string::size_type pos = path.rfind('.');
		std::string ext = path.substr(pos == std::string::npos ? path.length() : pos + 1);
		if (ext == "mat")
			return materialIm;
		else if (ext == "fbx" || ext == "FBX")
			return meshIm;
		else if (ext == "png" || ext == "jpg" || ext == "jpeg")
			return imageIm;
		else if (ext == "scene")
			return sceneIm;
		else if (ext == "glsl" || ext == "hlsl")
			return shaderIm;

		return fileIm;
	}
	
	auto ContentBrowser::onDrawGui() noexcept -> void
	{
		ImGui::Begin("Content Browser");

		// Return Button
		static std::filesystem::path root("./assets");
		if (root.compare(currentDirectory))
		{
			if (ImGui::Button("<-"))
			{
				currentDirectory = currentDirectory.parent_path();
			}
		}

		// calculate column number
		static float padding = 16.0f * dpi;
		static float thumbnailSize = 64.f * dpi;
		float cellSize = thumbnailSize + padding;
		float panelWidth = ImGui::GetContentRegionAvail().x;
		int columnCount = (int)(panelWidth / cellSize);
		if (columnCount < 1)
			columnCount = 1;
		// Do Columns
		ImGui::Columns(columnCount, 0, false);

		// for each subdir
		for (auto& directoryEntry : std::filesystem::directory_iterator(currentDirectory))
		{
			const auto& path = directoryEntry.path();
			std::string pathString = path.string();
			auto relativePath = std::filesystem::relative(directoryEntry.path(), root);
			std::string relativePathString = relativePath.string();
			std::string filenameString = relativePath.filename().string();

			if (filenameString == ".adb" || filenameString == ".iadb")
				continue;

			ImGui::PushID(filenameString.c_str());
			// If is directory
			ImImage* icon = directoryEntry.is_directory() ? folderIm : getIcon(filenameString);

			ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
			ImGui::ImageButton(icon->getImTextureID(), { thumbnailSize,thumbnailSize }, { 0,0 }, { 1,1 });

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
					currentDirectory /= path.filename();
				}
			}

		//	if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Left))
		//	{
		//		AssetType type = GetAssetType(filenameString);
		//		switch (type)
		//		{
		//		case SIByL::AssetType::Unknown:
		//			break;
		//		case SIByL::AssetType::Material:
		//		{
		//			Ref<Material> mat = GetAssetByPath<Material>(relativePathString);
		//			SIByLEditor::EditorLayer::s_InspectorPanel.SetSelectedMaterial(mat);
		//			break;
		//		}
		//		default:
		//			break;
		//		}
		//	}

			ImGui::TextWrapped(filenameString.c_str());

			ImGui::NextColumn();
			ImGui::PopID();
		}

		//////////////////////////////////////////////////////////////////////////////
		////							   Add Material Tab						      //
		//////////////////////////////////////////////////////////////////////////////
		//static bool NewMaterial = false;
		//// Right-click on blank space
		//if (ImGui::BeginPopupContextWindow(0, 1, false))
		//{
		//	if (ImGui::MenuItem("Create New Material"))
		//	{
		//		NewMaterial = true;
		//	}

		//	ImGui::EndPopup();
		//}

		//if (NewMaterial)
		//{
		//	Ref<Texture2D> icon = SIByLEditor::EditorLayer::IconMaterial;

		//	ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
		//	ImGui::ImageButton(icon->GetImGuiHandle(), { thumbnailSize,thumbnailSize }, { 0,1 }, { 1,0 });
		//	ImGui::PopStyleColor();

		//	//ImGui::ShowDemoWindow
		//	static char buf[32] = "DefaultMaterial";
		//	//static char buf[32] = u8"NIHONGO"; // <- this is how you would write it with C++11, using real kanjis
		//	if (ImGui::InputText(" ", buf, IM_ARRAYSIZE(buf), ImGuiInputTextFlags_EnterReturnsTrue | ImGuiInputTextFlags_AutoSelectAll))
		//	{
		//		NewMaterial = false;
		//		Ref<Material> newMat = CreateRef<Material>();
		//		MaterialSerializer matSerializer(newMat);
		//		std::string fullPath = m_CurrentDirectory.string() + "/" + std::string(buf) + ".mat";
		//		matSerializer.Serialize(fullPath);
		//	}

		//	//ImGui::TextWrapped(filenameString.c_str());
		//}

		ImGui::End();
	}
}