module;
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <imgui.h>
#include <imgui_internal.h>
#include "entt.hpp"
#include <map>
export module Editor.Introspection;
import Editor.Widget;
import Core.Application;
import Core.Layer;
import Core.LayerStack;

import Editor.Inspector;

import Asset.Asset;
import Asset.RuntimeAssetManager;
import Asset.AssetLayer;

namespace SIByL::Editor
{
	export struct Introspection :public Widget
	{
		virtual auto onDrawGui() noexcept -> void override;
		
		auto bindInspector(Inspector* inspector) noexcept -> void;
		auto bindApplication(IApplication* application) noexcept -> void;

	private:
		IApplication* application = nullptr;
		Inspector* inspector = nullptr;

		auto callbackDraw_AssetLayer_RuntimeAssetManager(Asset::RuntimeAssetManager* manager) -> void;

		auto onDrawGuiBranch_AssetLayer(Asset::AssetLayer* layer) noexcept -> void;
	};

	auto Introspection::bindInspector(Inspector* inspector) noexcept -> void
	{
		this->inspector = inspector;
	}
	auto Introspection::bindApplication(IApplication* application) noexcept -> void
	{
		this->application = application;
	}

	auto Introspection::onDrawGui() noexcept -> void
	{
		ImGui::Begin("Introspection", 0, ImGuiWindowFlags_MenuBar);

		if (application)
		{
			std::vector<ILayer*>& layer_stack = application->getLayerStack()->layer_stack;
			for (auto layer : layer_stack)
			{
				std::string name(layer->getName().data());
				if (ImGui::TreeNode(name.c_str()))
				{
					ImGui::NextColumn();
					if (name == "Asset Layer")
						onDrawGuiBranch_AssetLayer((Asset::AssetLayer*)layer);
					ImGui::TreePop();
				}
			}
		}

		ImGui::End();
	}

	auto Introspection::callbackDraw_AssetLayer_RuntimeAssetManager(Asset::RuntimeAssetManager* manager) -> void
	{
		static bool enable_texture = true;
		static bool enable_mesh = true;

		const ImGuiTreeNodeFlags treeNodeFlags =
			ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_FramePadding |
			ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap;
		static ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit;

		ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4,4 });
		float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		ImGui::Separator();
		bool open = ImGui::TreeNodeEx((void*)"RuntimeAssetManager", treeNodeFlags, "RuntimeAssetManager");
		ImGui::PopStyleVar();

		if (open)
		{
			ImGui::Checkbox("Texture", &enable_texture);
			ImGui::SameLine();
			ImGui::Checkbox("Mesh", &enable_mesh);

			ImGui::NextColumn();
			ImGui::Separator();
			ImGui::NextColumn();

			std::map<Asset::GUID, Asset::ResourceItem> const& assetMap = manager->getAssetMap();
			for (auto pair : assetMap)
			{
				Asset::GUID const& guid = pair.first;
				Asset::ResourceItem const& resource_item = pair.second;
				std::string guid_str = std::to_string(guid);
				if (ImGui::TreeNode(guid_str.c_str()))
				{
					ImGui::NextColumn();
					if (ImGui::BeginTable(guid_str.c_str(), 2, flags))
					{
						//ImGui::TableSetupColumn("key", 0, 100.0f);
						//ImGui::TableSetupColumn("value", 1);
						ImGui::TableNextRow();
						ImGui::TableSetColumnIndex(0);
						ImGui::Text("path");
						ImGui::TableSetColumnIndex(1);
						ImGui::Text(resource_item.path.c_str());
						ImGui::EndTable();
					}
					//ImGui::Separator();
					//ImGui::LabelText("Label", "Value");
					//ImGui::Text("blah blah");
					//ImGui::SameLine();

					ImGui::TreePop();
				}
			}

			ImGui::TreePop();
		}
	}

	auto Introspection::onDrawGuiBranch_AssetLayer(Asset::AssetLayer* layer) noexcept -> void
	{
		static ImGuiTreeNodeFlags base_flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick | ImGuiTreeNodeFlags_SpanAvailWidth;
		static ImGuiTreeNodeFlags bullet_flags = base_flags | ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen; // ImGuiTreeNodeFlags_Bullet
		bool node_open = ImGui::TreeNodeEx((void*)(intptr_t)0, bullet_flags, "Runtime Asset Manager", 0);
		if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen())
		{
			if(inspector) inspector->setCustomDraw(std::bind(&Introspection::callbackDraw_AssetLayer_RuntimeAssetManager, this, &(layer->runtimeManager)));
		}

	}
}