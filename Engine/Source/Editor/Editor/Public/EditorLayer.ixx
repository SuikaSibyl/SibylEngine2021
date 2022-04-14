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
import Editor.Scene;
import Editor.Inspector;
import Editor.Introspection;
import Editor.RenderPipeline;
import Editor.Viewport;

namespace SIByL::Editor
{
	export struct EditorLayer :public ILayer
	{
		EditorLayer() :ILayer("Editor Layer") {}
		auto onDrawGui() noexcept -> void;

		Viewport mainViewport;
		Scene sceneGui;
		Inspector inspectorGui;
		Introspection introspectionGui;
		RenderPipeline pipelineGui;
	};

	auto EditorLayer::onDrawGui() noexcept -> void
	{
		sceneGui.onDrawGui();
		pipelineGui.onDrawGui();
		introspectionGui.onDrawGui();
		inspectorGui.onDrawGui();

		mainViewport.onDrawGui();
	}

}