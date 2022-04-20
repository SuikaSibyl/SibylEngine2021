module;
#include <string>
#include <vector>
#include "spirv_glsl.hpp"
#include "spirv_cross.hpp"
module RHI.IShaderReflection.VK;
import RHI.IShaderReflection;
import RHI.IEnum;

namespace SIByL::RHI
{
	IShaderReflectionVK::IShaderReflectionVK(char const* code, size_t size, ShaderStageFlagBits stage)
	{
		std::vector<uint32_t> spirv_binary(size / sizeof(uint32_t));
		memcpy(spirv_binary.data(), code, size);
		spirv_cross::CompilerGLSL glsl(std::move(spirv_binary));
		// The SPIR-V is now parsed, and we can perform reflection on it.
		spirv_cross::ShaderResources resources = glsl.get_shader_resources();
		// Get all sampled images in the shader.
		for (auto& resource : resources.sampled_images)
		{
			unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
			descriptorItems.emplace_back(set, binding, RHI::DescriptorType::SAMPLER, resource.name, (uint32_t)stage);
		}
		// Get all uniform buffers in the shader.
		for (auto& resource : resources.uniform_buffers)
		{
			unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
			descriptorItems.emplace_back(set, binding, RHI::DescriptorType::UNIFORM_BUFFER, resource.name, (uint32_t)stage);
		}
		// Get all storage buffers in the shader.
		for (auto& resource : resources.storage_buffers)
		{
			unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
			descriptorItems.emplace_back(set, binding, RHI::DescriptorType::STORAGE_BUFFER, resource.name, (uint32_t)stage);
		}
		// Get all storage inputs in the shader.
		for (auto& resource : resources.storage_images)
		{
			unsigned set = glsl.get_decoration(resource.id, spv::DecorationDescriptorSet);
			unsigned binding = glsl.get_decoration(resource.id, spv::DecorationBinding);
			descriptorItems.emplace_back(set, binding, RHI::DescriptorType::STORAGE_IMAGE, resource.name, (uint32_t)stage);
		}
		// Get all push constant buffers inputs in the shader.
		for (auto& resource : resources.push_constant_buffers)
		{
			auto ranges = glsl.get_active_buffer_ranges(resource.id);
			for (auto& range : ranges)
			{
				pushConstantItems.emplace_back(range.index, range.offset, range.range, (uint32_t)stage);
			}
		}

		sortDescriptorItems();
		sortPushConstantItems();
	}
}