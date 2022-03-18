module;
#include <iostream>
#include <cstdint>
#include <imgui.h>
#include <vulkan/vulkan.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
export module Editor.ImGuiBackend.VK;
import Editor.ImGuiLayer;
import Core.Layer;
import Core.Log;
import Core.Window;
import Core.Event;
import Core.Window.GLFW;
import Core.MemoryManager;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
import RHI.IPhysicalDevice.VK;
import RHI.GraphicContext.VK;

namespace SIByL::Editor
{
    void check_vk_result(VkResult err)
    {
        if (err == 0)
            return;
        SE_CORE_ERROR("ImGui Vulkan Error: VkResult = {0}", (unsigned int)err);
        if (err < 0)
            return;
    }
    
    VkResult err;

	export struct ImGuiBackendVK :public ImGuiBackend
	{
        ImGuiBackendVK(uint32_t const& width, uint32_t const& height, RHI::ILogicalDevice* logical_device);

		virtual auto setupPlatformBackend() noexcept -> void override;
        virtual auto uploadFonts() noexcept -> void override;
        virtual auto onWindowResize(WindowResizeEvent& e) -> void override;

        virtual auto startNewFrame() -> void override;
        virtual auto render(ImDrawData* draw_data) -> void override;
        virtual auto present() -> void override;

        RHI::IGraphicContextVK* graphicContext;
        RHI::IPhysicalDeviceVK* physicalDevice;
        RHI::ILogicalDeviceVK* logicalDevice;
        IWindowGLFW* window;

		ImGui_ImplVulkanH_Window mainWindowData;
        VkPipelineCache pipelineCache = VK_NULL_HANDLE;
        VkDescriptorPool descriptorPool;
	};

    int g_MinImageCount = 2;

    ImGuiBackendVK::ImGuiBackendVK(uint32_t const& width, uint32_t const& height, RHI::ILogicalDevice* logical_device)
    {
        logicalDevice = reinterpret_cast<RHI::ILogicalDeviceVK*>(logical_device);
        physicalDevice = logicalDevice->getPhysicalDeviceVk();
        graphicContext = reinterpret_cast<RHI::IGraphicContextVK*>(physicalDevice->getGraphicContext());
        window = graphicContext->getAttachedWindow();

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

        // Create Descriptor Pool
        {
            VkDescriptorPoolSize pool_sizes[] =
            {
                { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
                { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
                { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
                { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
                { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
                { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
            };
            VkDescriptorPoolCreateInfo pool_info = {};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            pool_info.maxSets = 1000 * IM_ARRAYSIZE(pool_sizes);
            pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
            pool_info.pPoolSizes = pool_sizes;
            vkCreateDescriptorPool(logicalDevice->getDeviceHandle(), &pool_info, nullptr, &descriptorPool);
        }
    }

	auto ImGuiBackendVK::setupPlatformBackend() noexcept -> void
	{
        RHI::IPhysicalDeviceVK::QueueFamilyIndices indices = physicalDevice->findQueueFamilies();

        ImGui_ImplGlfw_InitForVulkan((GLFWwindow*)window->getNativeWindow(), true);
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = RHI::IGraphicContextVK::getVKInstance();
        init_info.PhysicalDevice = physicalDevice->getPhysicalDevice();
        init_info.Device = logicalDevice->getDeviceHandle();
        init_info.QueueFamily = indices.graphicsFamily.value();
        init_info.Queue = *logicalDevice->getVkGraphicQueue();
        init_info.PipelineCache = pipelineCache;
        init_info.DescriptorPool = descriptorPool;
        init_info.Subpass = 0;
        init_info.MinImageCount = g_MinImageCount;
        init_info.ImageCount = mainWindowData.ImageCount;
        init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        init_info.Allocator = nullptr;
        init_info.CheckVkResultFn = nullptr;
        ImGui_ImplVulkan_Init(&init_info, mainWindowData.RenderPass);
	}

    // Upload Fonts
    auto ImGuiBackendVK::uploadFonts() noexcept -> void
    {
        // Use any command queue
        VkCommandPool command_pool = mainWindowData.Frames[mainWindowData.FrameIndex].CommandPool;
        VkCommandBuffer command_buffer = mainWindowData.Frames[mainWindowData.FrameIndex].CommandBuffer;

        err = vkResetCommandPool(logicalDevice->getDeviceHandle(), command_pool, 0);
        check_vk_result(err);
        VkCommandBufferBeginInfo begin_info = {};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        err = vkBeginCommandBuffer(command_buffer, &begin_info);
        check_vk_result(err);

        ImGui_ImplVulkan_CreateFontsTexture(command_buffer);

        VkSubmitInfo end_info = {};
        end_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        end_info.commandBufferCount = 1;
        end_info.pCommandBuffers = &command_buffer;
        err = vkEndCommandBuffer(command_buffer);
        check_vk_result(err);
        err = vkQueueSubmit(*logicalDevice->getVkGraphicQueue(), 1, &end_info, VK_NULL_HANDLE);
        check_vk_result(err);

        err = vkDeviceWaitIdle(logicalDevice->getDeviceHandle());
        check_vk_result(err);
        ImGui_ImplVulkan_DestroyFontUploadObjects();
    }

    auto ImGuiBackendVK::onWindowResize(WindowResizeEvent& e) -> void
    {
        RHI::IPhysicalDeviceVK::QueueFamilyIndices indices = physicalDevice->findQueueFamilies();

        ImGui_ImplVulkan_SetMinImageCount(g_MinImageCount);
        ImGui_ImplVulkanH_CreateOrResizeWindow(
            RHI::IGraphicContextVK::getVKInstance(),
            physicalDevice->getPhysicalDevice(),
            logicalDevice->getDeviceHandle(),
            &mainWindowData,
            indices.graphicsFamily.value(),
            nullptr,
            e.GetWidth(), e.GetHeight(),
            g_MinImageCount); 
        mainWindowData.FrameIndex = 0;
    }

    auto ImGuiBackendVK::startNewFrame() -> void
    {
        // Start the Dear ImGui frame
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
    }

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    auto ImGuiBackendVK::render(ImDrawData* draw_data) -> void
    {
        mainWindowData.ClearValue.color.float32[0] = clear_color.x * clear_color.w;
        mainWindowData.ClearValue.color.float32[1] = clear_color.y * clear_color.w;
        mainWindowData.ClearValue.color.float32[2] = clear_color.z * clear_color.w;
        mainWindowData.ClearValue.color.float32[3] = clear_color.w;

        VkResult err;
        ImGui_ImplVulkanH_Window* wd = &mainWindowData;

        VkSemaphore image_acquired_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
        VkSemaphore render_complete_semaphore = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
        err = vkAcquireNextImageKHR(logicalDevice->getDeviceHandle(), wd->Swapchain, UINT64_MAX, image_acquired_semaphore, VK_NULL_HANDLE, &wd->FrameIndex);
        check_vk_result(err);

        ImGui_ImplVulkanH_Frame* fd = &wd->Frames[wd->FrameIndex];
        {
            err = vkWaitForFences(logicalDevice->getDeviceHandle(), 1, &fd->Fence, VK_TRUE, UINT64_MAX);    // wait indefinitely instead of periodically checking
            check_vk_result(err);

            err = vkResetFences(logicalDevice->getDeviceHandle(), 1, &fd->Fence);
            check_vk_result(err);
        }
        {
            err = vkResetCommandPool(logicalDevice->getDeviceHandle(), fd->CommandPool, 0);
            check_vk_result(err);
            VkCommandBufferBeginInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            info.flags |= VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            err = vkBeginCommandBuffer(fd->CommandBuffer, &info);
            check_vk_result(err);
        }
        {
            VkRenderPassBeginInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            info.renderPass = wd->RenderPass;
            info.framebuffer = fd->Framebuffer;
            info.renderArea.extent.width = wd->Width;
            info.renderArea.extent.height = wd->Height;
            info.clearValueCount = 1;
            info.pClearValues = &wd->ClearValue;
            vkCmdBeginRenderPass(fd->CommandBuffer, &info, VK_SUBPASS_CONTENTS_INLINE);
        }

        // Record dear imgui primitives into command buffer
        ImGui_ImplVulkan_RenderDrawData(draw_data, fd->CommandBuffer);

        // Submit command buffer
        vkCmdEndRenderPass(fd->CommandBuffer);
        {
            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            VkSubmitInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            info.waitSemaphoreCount = 1;
            info.pWaitSemaphores = &image_acquired_semaphore;
            info.pWaitDstStageMask = &wait_stage;
            info.commandBufferCount = 1;
            info.pCommandBuffers = &fd->CommandBuffer;
            info.signalSemaphoreCount = 1;
            info.pSignalSemaphores = &render_complete_semaphore;

            err = vkEndCommandBuffer(fd->CommandBuffer);
            check_vk_result(err);
            err = vkQueueSubmit(*logicalDevice->getVkGraphicQueue(), 1, &info, fd->Fence);
            check_vk_result(err);
        }
    }

    auto ImGuiBackendVK::present() -> void
    {
        VkSemaphore render_complete_semaphore = mainWindowData.FrameSemaphores[mainWindowData.SemaphoreIndex].RenderCompleteSemaphore;
        VkPresentInfoKHR info = {};
        info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &render_complete_semaphore;
        info.swapchainCount = 1;
        info.pSwapchains = &mainWindowData.Swapchain;
        info.pImageIndices = &mainWindowData.FrameIndex;
        VkResult err = vkQueuePresentKHR(*logicalDevice->getVkGraphicQueue(), &info);
        check_vk_result(err);
        mainWindowData.SemaphoreIndex = (mainWindowData.SemaphoreIndex + 1) % mainWindowData.ImageCount; // Now we can use the next set of semaphores
    }
}