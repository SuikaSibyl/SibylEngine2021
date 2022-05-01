module;
#include <vulkan/vulkan.h>
#include <vector>
#include <optional>
export module RHI.ILogicalDevice.VK;
import RHI.IBarrier;
import RHI.ILogicalDevice;
import RHI.IPhysicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class ILogicalDeviceVK :public ILogicalDevice
		{
		public:
			ILogicalDeviceVK(IPhysicalDeviceVK* physicalDevice);
			virtual ~ILogicalDeviceVK() = default;

			virtual auto initialize() -> bool;
			virtual auto destroy() -> bool;
			virtual auto getPhysicalDevice() noexcept -> IPhysicalDevice* override;
			virtual auto waitIdle() noexcept -> void override;

			auto getDeviceHandle() noexcept -> VkDevice&;
			auto getPhysicalDeviceVk() noexcept -> IPhysicalDeviceVK*;
			auto getVkGraphicQueue() noexcept -> VkQueue*;
			auto getVkPresentQueue() noexcept -> VkQueue*;
			auto getVkComputeQueue() noexcept -> VkQueue*;

			auto allocMemory(
				VkMemoryRequirements* memRequirements,
				VkBuffer* vertexBuffer,
				VkDeviceMemory* vertexBufferMemory) noexcept -> void;

			virtual auto getRasterStageMask() noexcept -> PipelineStageFlags override { return rasterStageMask; }

		private:
			IPhysicalDeviceVK* physicalDevice;
			VkDevice device;
			VkQueue graphicsQueue;
			VkQueue presentQueue;
			VkQueue computeQueue;

		private:
			RHI::PipelineStageFlags rasterStageMask = 
				(uint32_t)RHI::PipelineStageFlagBits::VERTEX_SHADER_BIT | 
				(uint32_t)RHI::PipelineStageFlagBits::FRAGMENT_SHADER_BIT;

			auto createLogicalDevice(IPhysicalDeviceVK* physicalDevice) noexcept -> void;
		};
	}
}