module;
#include <vulkan/vulkan.h>
export module RHI.IPipelineLayout.VK;
import RHI.IPipelineLayout;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IPipelineLayoutVK :public IPipelineLayout
		{
		public:
			IPipelineLayoutVK() = default;
			IPipelineLayoutVK(IPipelineLayoutVK&&) = default;
			virtual ~IPipelineLayoutVK();

			auto getVkPipelineLayout() noexcept -> VkPipelineLayout*;
			auto createPipelineLayout() noexcept -> void;
		private:
			ILogicalDeviceVK* logicalDevice;
			VkPipelineLayout pipelineLayout;
		};
	}
}
