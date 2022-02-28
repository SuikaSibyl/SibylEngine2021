module;
#include <cstdint>
#include <vulkan/vulkan.h>
#include <vector>
module RHI.IDescriptorPool.VK;
import Core.Log;
import RHI.IEnum;
import RHI.IEnum.VK;
import RHI.IDescriptorPool;
import RHI.ILogicalDevice.VK;

namespace SIByL::RHI
{
	void createDescriptorPool(
		DescriptorPoolDesc const& desc,
		VkDescriptorPool* descriptor_pool,
		ILogicalDeviceVK* logical_device)
	{
		std::vector<std::pair<DescriptorType, uint32_t>> const& type_and_count = desc.typeAndCount;
		uint32_t const& max_sets = desc.max_sets;

		std::vector<VkDescriptorPoolSize> descriptors(type_and_count.size());
		unsigned int idx = 0;
		for (auto& pair : type_and_count)
		{
			descriptors[idx].type = getVkDescriptorType(pair.first);
			descriptors[idx].descriptorCount = pair.second;
			idx++;
		}
		
		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.poolSizeCount = descriptors.size();
		poolInfo.pPoolSizes = descriptors.data();
		poolInfo.maxSets = static_cast<uint32_t>(max_sets);

		if (vkCreateDescriptorPool(logical_device->getDeviceHandle(), &poolInfo, nullptr, descriptor_pool) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to create descriptor pool!");
		}
	}

	IDescriptorPoolVK::IDescriptorPoolVK(DescriptorPoolDesc const& desc, ILogicalDeviceVK* logical_device)
		: logicalDevice(logical_device)
	{
		createDescriptorPool(desc, &descriptorPool, logicalDevice);
	}

	IDescriptorPoolVK::~IDescriptorPoolVK()
	{
		vkDestroyDescriptorPool(logicalDevice->getDeviceHandle(), descriptorPool, nullptr);
	}

	auto IDescriptorPoolVK::getVkDescriptorPool() noexcept -> VkDescriptorPool*
	{
		return &descriptorPool;
	}
}
