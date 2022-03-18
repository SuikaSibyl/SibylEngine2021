module;
#include <imgui.h>
#include <utility>
#include <type_traits>
module Editor.ImGuiLayer;
import Core.Layer;
import Core.Log;
import Core.MemoryManager;
import RHI.IEnum;
import RHI.ILogicalDevice;
import Editor.ImGuiBackend.VK;

namespace SIByL::Editor
{
	ImGuiLayer::ImGuiLayer(RHI::ILogicalDevice* logical_device)
	{
		switch (logical_device->getPhysicalDevice()->getGraphicContext()->getAPI())
		{
		case RHI::API::VULKAN:
			{
            MemScope<ImGuiBackendVK> backend_vk = MemNew<ImGuiBackendVK>(1280, 720, logical_device);
            backend = MemCast<ImGuiBackend>(backend_vk);
			}
			break;
		default:
            SE_CORE_ERROR("Editor Layer :: Get Invalid API graphic context");
            break;
		}

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows
        //io.ConfigViewportsNoAutoMerge = true;
		//io.ConfigViewportsNoTaskBarIcon = true;

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();
        
		// When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
        ImGuiStyle& style = ImGui::GetStyle();
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            style.WindowRounding = 0.0f;
            style.Colors[ImGuiCol_WindowBg].w = 1.0f;
        }


    }
}