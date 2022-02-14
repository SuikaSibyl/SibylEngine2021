module;
#include <vulkan/vulkan.h>
export module RHI.IShader.VK;
import Core.SObject;
import RHI.IShader;
import RHI.ILogicalDevice.VK;
import RHI.IEnum;
import RHI.IEnum.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IShaderVK : public IShader
		{
		public:
			virtual ~IShaderVK();

			auto getVkShaderModule() noexcept -> VkShaderModule&;
			auto createShaderModule(char const* code, size_t size) noexcept -> void;
			auto getVkShaderStageCreateInfo() noexcept -> VkPipelineShaderStageCreateInfo*;
			auto createVkShaderStageCreateInfo() noexcept -> void;

		private:
			ILogicalDeviceVK* logicalDevice;
			VkShaderModule shaderModule;
			VkPipelineShaderStageCreateInfo shaderStageInfo{};
		};
	}
}