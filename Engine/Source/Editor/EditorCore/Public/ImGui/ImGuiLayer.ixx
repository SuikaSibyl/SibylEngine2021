module;
#include <imgui.h>
export module Editor.ImGuiLayer;
import Core.Layer;
import Core.MemoryManager;
import RHI.ILogicalDevice;

namespace SIByL::Editor
{
	export struct ImGuiBackend
	{
		virtual auto setupPlatformBackend() noexcept -> void = 0;
	};

	export class ImGuiLayer :public ILayer
	{
	public:
		ImGuiLayer(RHI::ILogicalDevice* logical_device);
		MemScope<ImGuiBackend> backend;
	};
}