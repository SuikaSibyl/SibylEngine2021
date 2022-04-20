module;
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cstdint>
#include <imgui.h>
#include <imgui_internal.h>
#include "entt.hpp"
export module Editor.EditorLayer;
import Core.Layer;
import Core.Time;
import Core.Window;
import Editor.Scene;
import Editor.Inspector;
import Editor.Introspection;
import Editor.RenderPipeline;
import Editor.Viewport;
import Editor.ContentBrowser;
import Asset.AssetLayer;
import Editor.ImGuiLayer;

namespace SIByL::Editor
{
	export struct EditorLayer :public ILayer
	{
		EditorLayer(WindowLayer* window_layer, Asset::AssetLayer* asset_layer, ImGuiLayer* imgui_layer, Timer* timer) :ILayer("Editor Layer"),
			mainViewport(window_layer, timer),
			sceneGui(window_layer, asset_layer),
			contentBrowserGui(window_layer, asset_layer, imgui_layer) {}
		auto onDrawGui() noexcept -> void;
		virtual auto onUpdate() -> void override;
		Viewport mainViewport;
		Scene sceneGui;
		Inspector inspectorGui;
		Introspection introspectionGui;
		RenderPipeline pipelineGui;
		ContentBrowser contentBrowserGui;
	};

	auto EditorLayer::onDrawGui() noexcept -> void
	{
		sceneGui.onDrawGui();
		pipelineGui.onDrawGui();
		contentBrowserGui.onDrawGui();
		introspectionGui.onDrawGui();
		inspectorGui.onDrawGui();

		mainViewport.onDrawGui();
	}
	
	auto EditorLayer::onUpdate() -> void
	{
		mainViewport.onUpdate();
	}

}