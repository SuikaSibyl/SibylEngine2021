module;
#include <cstdint>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
export module Editor.ImGuiBackend.VK;
import Editor.ImGuiLayer;
import Core.Layer;
import Core.MemoryManager;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
import RHI.IPhysicalDevice.VK;
import RHI.GraphicContext.VK;

namespace SIByL::Editor
{
	export struct ImGuiBackendVK :public ImGuiBackend
	{
        ImGuiBackendVK(uint32_t const& width, uint32_t const& height, RHI::ILogicalDevice* logical_device);

		virtual auto setupPlatformBackend() noexcept -> void override;

        RHI::IGraphicContextVK* graphicContext;
        RHI::IPhysicalDeviceVK* physicalDevice;
        RHI::ILogicalDeviceVK* logicalDevice;

		ImGui_ImplVulkanH_Window mainWindowData;
	};

    int g_MinImageCount = 2;

    ImGuiBackendVK::ImGuiBackendVK(uint32_t const& width, uint32_t const& height, RHI::ILogicalDevice* logical_device)
    {
        logicalDevice = reinterpret_cast<RHI::ILogicalDeviceVK*>(logical_device);
        physicalDevice = logicalDevice->getPhysicalDeviceVk();
        graphicContext = reinterpret_cast<RHI::IGraphicContextVK*>(physicalDevice->getGraphicContext());

        mainWindowData.Surface = graphicContext->surface;

        // Select Surface Format
        const VkFormat requestSurfaceImageFormat[] = { VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8_UNORM, VK_FORMAT_R8G8B8_UNORM };
        const VkColorSpaceKHR requestSurfaceColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
        mainWindowData.SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
            physicalDevice->getPhysicalDevice(), 
            mainWindowData.Surface, 
            requestSurfaceImageFormat, 
            (size_t)IM_ARRAYSIZE(requestSurfaceImageFormat), 
            requestSurfaceColorSpace);

        VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
        mainWindowData.PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
            physicalDevice->getPhysicalDevice(), 
            mainWindowData.Surface,
            &present_modes[0], 
            IM_ARRAYSIZE(present_modes));
        
        //printf("[vulkan] Selected PresentMode = %d\n", wd->PresentMode);
        RHI::IPhysicalDeviceVK::QueueFamilyIndices indices = physicalDevice->findQueueFamilies();

        // Create SwapChain, RenderPass, Framebuffer, etc.
        IM_ASSERT(g_MinImageCount >= 2);
        ImGui_ImplVulkanH_CreateOrResizeWindow(
            RHI::IGraphicContextVK::getVKInstance(),
            physicalDevice->getPhysicalDevice(), 
            logicalDevice->getDeviceHandle(),
            &mainWindowData, 
            indices.graphicsFamily.value(),
            nullptr, 
            width,
            height,
            g_MinImageCount);
    }

	auto ImGuiBackendVK::setupPlatformBackend() noexcept -> void
	{
        //ImGui_ImplGlfw_InitForVulkan(window, true);
        //ImGui_ImplVulkan_InitInfo init_info = {};
        //init_info.Instance = g_Instance;
        //init_info.PhysicalDevice = g_PhysicalDevice;
        //init_info.Device = g_Device;
        //init_info.QueueFamily = g_QueueFamily;
        //init_info.Queue = g_Queue;
        //init_info.PipelineCache = g_PipelineCache;
        //init_info.DescriptorPool = g_DescriptorPool;
        //init_info.Subpass = 0;
        //init_info.MinImageCount = g_MinImageCount;
        //init_info.ImageCount = wd->ImageCount;
        //init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        //init_info.Allocator = g_Allocator;
        //init_info.CheckVkResultFn = check_vk_result;
        //ImGui_ImplVulkan_Init(&init_info, wd->RenderPass);
	}
}