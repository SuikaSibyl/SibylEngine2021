module;
#include <vulkan/vulkan.h>
module RHI.IDescriptorSet.VK;
import Core.Log;
import RHI.IDescriptorSet;
import RHI.ILogicalDevice.VK;
import RHI.IDescriptorPool;
import RHI.IDescriptorPool.VK;
import RHI.IDescriptorSetLayout;
import RHI.IDescriptorSetLayout.VK;
import RHI.IUniformBuffer;
import RHI.IUniformBuffer.VK;
import RHI.ITextureView;
import RHI.ITextureView.VK;
import RHI.ISampler;
import RHI.ISampler.VK;
import RHI.IStorageBuffer;
import RHI.IStorageBuffer.VK;

namespace SIByL::RHI
{
	auto createDescriptorSet(
		IDescriptorPoolVK* descriptor_pool,
		VkDescriptorSetLayout* layout,
		VkDescriptorSet* set,
		ILogicalDeviceVK* logical_device
	) noexcept -> void
	{
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = *descriptor_pool->getVkDescriptorPool();
		allocInfo.descriptorSetCount = 1;
		allocInfo.pSetLayouts = layout;

		if (vkAllocateDescriptorSets(logical_device->getDeviceHandle(), &allocInfo, set) != VK_SUCCESS) {
			SE_CORE_ERROR("VULKAN :: failed to allocate descriptor sets!");
		}
	}

	IDescriptorSetVK::IDescriptorSetVK(DescriptorSetDesc const& desc, ILogicalDeviceVK* logical_device)
		: logicalDevice(logical_device)
		, descriptorPool((IDescriptorPoolVK*)desc.descriptorPool)
	{
		createDescriptorSet(
			(IDescriptorPoolVK*)desc.descriptorPool,
			((IDescriptorSetLayoutVK*)desc.layout)->getVkDescriptorSetLayout(),
			&set,
			logicalDevice);
	}

	IDescriptorSetVK::~IDescriptorSetVK()
	{
		//vkFreeDescriptorSets(
		//	logicalDevice->getDeviceHandle(),
		//	*(descriptorPool)->getVkDescriptorPool(),
		//	1,
		//	&set);
	}

	auto IDescriptorSetVK::update(IUniformBuffer* uniform_buffer, uint32_t const& binding, uint32_t const& array_element) noexcept -> void
	{
		IUniformBufferVK* uniform_buffer_vk = (IUniformBufferVK*)uniform_buffer;

		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = *uniform_buffer_vk->getVkBuffer();
		bufferInfo.offset = 0;
		bufferInfo.range = uniform_buffer_vk->getSize();

		VkWriteDescriptorSet descriptorWrite{};
		descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet = set;
		descriptorWrite.dstBinding = binding;
		descriptorWrite.dstArrayElement = array_element;
		descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pBufferInfo = &bufferInfo;
		descriptorWrite.pImageInfo = nullptr; // Optional
		descriptorWrite.pTexelBufferView = nullptr; // Optional

		vkUpdateDescriptorSets(logicalDevice->getDeviceHandle(), 1, &descriptorWrite, 0, nullptr);
	}

	auto IDescriptorSetVK::update(IStorageBuffer* storage_buffer, uint32_t const& binding, uint32_t const& array_element) noexcept -> void
	{
		IStorageBufferVK* storage_buffer_VK = (IStorageBufferVK*)storage_buffer;

		VkDescriptorBufferInfo bufferInfo{};
		bufferInfo.buffer = *storage_buffer_VK->getVkBuffer();
		bufferInfo.offset = 0;
		bufferInfo.range = storage_buffer_VK->getSize();

		VkWriteDescriptorSet descriptorWrite{};
		descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet = set;
		descriptorWrite.dstBinding = binding;
		descriptorWrite.dstArrayElement = array_element;
		descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pBufferInfo = &bufferInfo;
		descriptorWrite.pImageInfo = nullptr; // Optional
		descriptorWrite.pTexelBufferView = nullptr; // Optional

		vkUpdateDescriptorSets(logicalDevice->getDeviceHandle(), 1, &descriptorWrite, 0, nullptr);
	}

	auto IDescriptorSetVK::update(ITextureView* texture_view, ISampler* sampler, uint32_t const& binding, uint32_t const& array_element) noexcept -> void
	{
		VkDescriptorImageInfo imageInfo{};
		imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		imageInfo.imageView = *((ITextureViewVK*)texture_view)->getpVkImageView();
		imageInfo.sampler = *((ISamplerVK*)sampler)->getVkSampler();
		
		VkWriteDescriptorSet descriptorWrite{};
		descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		descriptorWrite.dstSet = set;
		descriptorWrite.dstBinding = binding;
		descriptorWrite.dstArrayElement = array_element;
		descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		descriptorWrite.descriptorCount = 1;
		descriptorWrite.pBufferInfo = nullptr;
		descriptorWrite.pImageInfo = &imageInfo; // Optional
		descriptorWrite.pTexelBufferView = nullptr; // Optional

		vkUpdateDescriptorSets(logicalDevice->getDeviceHandle(), 1, &descriptorWrite, 0, nullptr);
	}

	auto IDescriptorSetVK::getVkDescriptorSet() noexcept -> VkDescriptorSet*
	{
		return &set;
	}
}
