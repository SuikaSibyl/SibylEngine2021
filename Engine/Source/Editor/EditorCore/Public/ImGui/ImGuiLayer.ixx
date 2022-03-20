module;
#include <imgui.h>
export module Editor.ImGuiLayer;
import Core.Layer;
import Core.Window;
import Core.Event;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.ILogicalDevice;

namespace SIByL::Editor
{
	export struct ImGuiBackend
	{
		virtual auto setupPlatformBackend() noexcept -> void = 0;
		virtual auto uploadFonts() noexcept -> void = 0;
		virtual auto getWindowDPI() noexcept -> float = 0;
		virtual auto onWindowResize(WindowResizeEvent& e) -> void = 0;

		virtual auto startNewFrame() -> void = 0;
		virtual auto render(ImDrawData* draw_data) -> void = 0;
		virtual auto present() -> void = 0;
	};

	export class ImGuiLayer :public ILayer
	{
	public:
		ImGuiLayer(RHI::ILogicalDevice* logical_device);
		auto onEvent(Event& e) -> void;
		auto onWindowResize(WindowResizeEvent& e) -> bool;

		auto startNewFrame() -> void;
		auto startGuiRecording() -> void;
		auto render() -> void;

		RHI::API api;
		MemScope<ImGuiBackend> backend;
	};
}