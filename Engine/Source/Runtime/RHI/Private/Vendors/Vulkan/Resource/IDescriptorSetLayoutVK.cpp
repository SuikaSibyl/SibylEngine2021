module;
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>
module RHI.IDescriptorSetLayout.VK;
import Core.Log;
import RHI.IEnum;
import RHI.IEnum.VK;
import RHI.IDescriptorSetLayout;
import RHI.ILogicalDevice.VK;
import RHI.ISampler;

namespace SIByL::RHI
{
    void createDescriptorSetLayout(
        DescriptorSetLayoutDesc const& desc,
        VkDescriptorSetLayout* layout,
        ILogicalDeviceVK* logical_device) 
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings(desc.perBindingDesc.size());
        for (int i = 0; i < desc.perBindingDesc.size(); i++)
        {
            bindings[i].binding = desc.perBindingDesc[i].binding;
            bindings[i].descriptorType = getVkDescriptorType(desc.perBindingDesc[i].type);
            bindings[i].descriptorCount = desc.perBindingDesc[i].count;
            bindings[i].stageFlags = getVkShaderStageFlags(desc.perBindingDesc[i].stages);
            bindings[i].pImmutableSamplers = nullptr; // Optional
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = bindings.size();
        layoutInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(logical_device->getDeviceHandle(), &layoutInfo, nullptr, layout) != VK_SUCCESS) {
            SE_CORE_ERROR("VULKAN :: failed to create descriptor set layout!");
        }
    }

    IDescriptorSetLayoutVK::IDescriptorSetLayoutVK(DescriptorSetLayoutDesc const& desc, ILogicalDeviceVK* logical_device)
        : logicalDevice(logical_device)
    {
        createDescriptorSetLayout(desc, &layout, logicalDevice);
    }

    IDescriptorSetLayoutVK::~IDescriptorSetLayoutVK()
    {
        vkDestroyDescriptorSetLayout(logicalDevice->getDeviceHandle(), layout, nullptr);
    }

    auto IDescriptorSetLayoutVK::getVkDescriptorSetLayout() noexcept -> VkDescriptorSetLayout*
    {
        return &layout;
    }
}
