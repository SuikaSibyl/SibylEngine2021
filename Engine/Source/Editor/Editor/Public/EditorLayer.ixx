module;
#include <Macros.h>
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
import Core.Event;
import Core.Window;
import GFX.RDG.RenderGraph;
import Editor.Scene;
import Editor.Inspector;
import Editor.Introspection;
import Editor.RenderPipeline;
import Editor.Viewport;
import Editor.ContentBrowser;
import Asset.AssetLayer;
import Editor.ImGuiLayer;
import Editor.RDGImImageManager;

namespace SIByL::Editor
{
	export struct EditorLayer :public ILayer
	{
		EditorLayer(WindowLayer* window_layer, Asset::AssetLayer* asset_layer, ImGuiLayer* imgui_layer, Timer* timer, GFX::RDG::RenderGraph* renderGraph) :ILayer("Editor Layer"),
			imImageManager(renderGraph, imgui_layer),
			mainViewport(window_layer, timer),
			sceneGui(window_layer, asset_layer, &mainViewport),
			contentBrowserGui(window_layer, asset_layer, imgui_layer) {}

		auto onDrawGui() noexcept -> void;
		virtual auto onUpdate() -> void override;
		virtual void onEvent(Event& event) override;
		auto onKeyPressedEvent(KeyPressedEvent& e) -> bool;

		RDGImImageManager imImageManager;

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

	void EditorLayer::onEvent(Event& e)
	{
		// application handling
		EventDispatcher dispatcher(e);
		dispatcher.Dispatch<KeyPressedEvent>(BIND_EVENT_FN(EditorLayer::onKeyPressedEvent));
	}

	auto EditorLayer::onKeyPressedEvent(KeyPressedEvent& e) -> bool
	{
		mainViewport.onKeyPressedEvent(e);
		return false;
	}
}