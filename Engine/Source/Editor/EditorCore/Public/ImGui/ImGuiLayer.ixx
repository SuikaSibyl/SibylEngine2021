module;
#include <imgui.h>
#include <unordered_map>
export module Editor.ImGuiLayer;
import Core.Layer;
import Core.Window;
import Core.Event;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.ILogicalDevice;
import RHI.IEnum;
import RHI.ISampler;
import RHI.ITextureView;
import Editor.ImImage;
import Asset.Asset;
import GFX.Texture;

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

	export struct ImImageLibrary
	{
		auto findImImage(Asset::GUID) noexcept -> ImImage*;
		auto addImImage(Asset::GUID, MemScope<ImImage>&& image) noexcept -> ImImage*;
		std::unordered_map<Asset::GUID, MemScope<ImImage>> imageLib;
	};

	export class ImGuiLayer :public ILayer
	{
	public:
		ImGuiLayer(RHI::ILogicalDevice* logical_device);
		auto onEvent(Event& e) -> void;
		auto onWindowResize(WindowResizeEvent& e) -> bool;

		auto createImImage(RHI::ISampler* sampler, RHI::ITextureView* view, RHI::ImageLayout layout) noexcept -> MemScope<ImImage>;
		auto getImImage(GFX::Texture const& texture) noexcept -> ImImage*;

		auto startNewFrame() -> void;
		auto startGuiRecording() -> void;
		auto render() -> void;

		RHI::API api;
		ImImageLibrary imImageLibrary;
		MemScope<RHI::ISampler> sampler;
		MemScope<ImGuiBackend> backend;
	};
}