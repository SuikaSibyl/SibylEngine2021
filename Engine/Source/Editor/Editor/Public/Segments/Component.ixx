module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <imgui_internal.h>
#include <functional>
#include "entt.hpp"
export module Editor.Component;
import Editor.Segment;

namespace SIByL::Editor
{
	export struct Component :public Segment
	{
		using CustomDrawFn = std::function<void()>;
		static auto onDrawGui(void* id, std::string const& name, CustomDrawFn open_fn, CustomDrawFn before_open_fn) noexcept -> void;
	};

	auto Component::onDrawGui(void* id, std::string const& name, CustomDrawFn open_fn, CustomDrawFn before_open_fn) noexcept -> void
	{
		const ImGuiTreeNodeFlags treeNodeFlags =
			ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_Framed | ImGuiTreeNodeFlags_FramePadding |
			ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_AllowItemOverlap;

		ImVec2 contentRegionAvailable = ImGui::GetContentRegionAvail();

		ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2{ 4,4 });
		float lineHeight = GImGui->Font->FontSize + GImGui->Style.FramePadding.y * 2.0f;
		ImGui::Separator();
		bool open = ImGui::TreeNodeEx(id, treeNodeFlags, name.c_str());
		ImGui::PopStyleVar();
		ImGui::SameLine(contentRegionAvailable.x - lineHeight * 0.5);
		if (ImGui::Button("+", ImVec2{ lineHeight, lineHeight }))
		{
			ImGui::OpenPopup("ComponentSettings");
		}
		
		if (before_open_fn) before_open_fn();
		if (open)
		{
			if (open_fn) open_fn();
			ImGui::TreePop();
		}
	}
}