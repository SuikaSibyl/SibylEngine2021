module;
#include <string>
#include <vulkan/vulkan.h>
export module RHI.IShader.VK;
import Core.SObject;
import RHI.IShader;
import RHI.IEnum;
import RHI.IEnum.VK;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IShaderVK : public IShader
		{
		public:
			IShaderVK(ILogicalDeviceVK* logical_device);
			virtual ~IShaderVK();

			virtual auto injectDesc(ShaderDesc const& desc) noexcept -> void override;

			auto getVkShaderModule() noexcept -> VkShaderModule&;
			auto createShaderModule(char const* code, size_t size) noexcept -> void;
			auto getVkShaderStageCreateInfo() noexcept -> VkPipelineShaderStageCreateInfo*;
			auto createVkShaderStageCreateInfo() noexcept -> void;

		private:
			ShaderStage stage;
			std::string entryPoint;
			ILogicalDeviceVK* logicalDevice;
			VkShaderModule shaderModule;
			VkPipelineShaderStageCreateInfo shaderStageInfo{};
		};
	}
}