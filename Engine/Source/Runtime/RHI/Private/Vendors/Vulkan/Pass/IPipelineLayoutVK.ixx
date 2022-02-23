module;
#include <vulkan/vulkan.h>
#include <vector>
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
			IPipelineLayoutVK(PipelineLayoutDesc const& desc, ILogicalDeviceVK* logical_device);
			IPipelineLayoutVK(IPipelineLayoutVK&&) = default;
			virtual ~IPipelineLayoutVK();

			auto getVkPipelineLayout() noexcept -> VkPipelineLayout*;
			auto createPipelineLayout(PipelineLayoutDesc const& desc) noexcept -> void;
		private:
			ILogicalDeviceVK* logicalDevice;
			VkPipelineLayout pipelineLayout;
		};
	}
}
