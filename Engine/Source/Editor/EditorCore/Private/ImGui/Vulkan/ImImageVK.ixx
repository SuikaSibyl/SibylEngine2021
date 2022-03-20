module;
#include <imgui.h>
#include <backends/imgui_impl_vulkan.h>
#include <vulkan/vulkan.h>
export module Editor.ImImage.VK;
import Editor.ImImage;
import RHI.IEnum;
import RHI.ISampler;
import RHI.ITextureView;
import RHI.IEnum.VK;
import RHI.ISampler.VK;
import RHI.ITextureView.VK;

namespace SIByL::Editor
{
	export struct ImImageVK :public ImImage
	{
		ImImageVK(RHI::ISampler* sampler, RHI::ITextureView* view, RHI::ImageLayout layout);
		virtual auto getImTextureID() noexcept -> ImTextureID override;

		VkDescriptorSet descriptorSet;
	};

	ImImageVK::ImImageVK(RHI::ISampler* sampler, RHI::ITextureView* view, RHI::ImageLayout layout)
	{
		descriptorSet = ImGui_ImplVulkan_AddTexture(
			*((RHI::ISamplerVK*)sampler)->getVkSampler(), 
			*((RHI::ITextureViewVK*)view)->getpVkImageView(), 
			getVkImageLayout(layout));
	}

	auto ImImageVK::getImTextureID() noexcept -> ImTextureID
	{
		return (ImTextureID)descriptorSet;
	}
}