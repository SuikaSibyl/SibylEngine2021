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
import Editor.Scene;
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

		auto bindScene(Editor::Scene* scene) { binded_scene = scene; }
		auto setCustomDraw(std::function<void()> func) noexcept -> void { customDraw = func; }
		auto kickInspectorEmpty() noexcept -> void;

		std::function<void()> customDraw;
		Editor::Scene* binded_scene;
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