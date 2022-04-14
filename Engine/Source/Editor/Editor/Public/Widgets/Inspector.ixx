module;
#include <vector>
#include <string>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <functional>
#include "entt.hpp"
export module Editor.Inspector;
import Editor.Widget;
import ECS.Entity;
import ECS.UID;
import ECS.TagComponent;
import GFX.SceneTree;
import GFX.Scene;

namespace SIByL::Editor
{
	export struct Inspector :public Widget
	{
		virtual auto onDrawGui() noexcept -> void override;

		auto setCustomDraw(std::function<void()> func) noexcept -> void { customDraw = func; }
		auto kickInspectorEmpty() noexcept -> void;

		std::function<void()> customDraw;
	};

	auto Inspector::onDrawGui() noexcept -> void
	{
		ImGui::Begin("Inspector", 0, ImGuiWindowFlags_MenuBar);
		if (customDraw) customDraw();
		ImGui::End();
	}

	auto Inspector::kickInspectorEmpty() noexcept -> void
	{
		setCustomDraw([]() {});
	}

}