module;
#include <string>
#include <vulkan/vulkan.h>
export module RHI.IShader.VK;
import Core.SObject;
import Core.MemoryManager;
import RHI.IShader;
import RHI.IEnum;
import RHI.IEnum.VK;
import RHI.ILogicalDevice.VK;
import RHI.IShaderReflection;
import RHI.IShaderReflection.VK;

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
			virtual auto getReflection() noexcept -> IShaderReflection * override;

			auto getVkShaderModule() noexcept -> VkShaderModule&;
			auto createShaderModule(char const* code, size_t size) noexcept -> void;
			auto getVkShaderStageCreateInfo() noexcept -> VkPipelineShaderStageCreateInfo*;
			auto createVkShaderStageCreateInfo() noexcept -> void;

		private:
			MemScope<IShaderReflectionVK> reflection;
			ShaderStage stage;
			std::string entryPoint;
			ILogicalDeviceVK* logicalDevice;
			VkShaderModule shaderModule;
			VkPipelineShaderStageCreateInfo shaderStageInfo{};
		};
	}
}