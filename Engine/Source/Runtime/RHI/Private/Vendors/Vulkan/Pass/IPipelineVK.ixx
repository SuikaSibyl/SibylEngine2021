module;
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IPipeline.VK;
import RHI.IPipeline;
import RHI.IResource;
import RHI.IShader;
import RHI.IFixedFunctions;
import RHI.IPipelineLayout;
import RHI.IRenderPass;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IPipelineVK :public IPipeline
		{
		public:
			IPipelineVK(PipelineDesc const& desc, ILogicalDeviceVK* logical_device);
			IPipelineVK(ComputePipelineDesc const& desc, ILogicalDeviceVK* logical_device);
			IPipelineVK(IPipelineVK&&) = default;
			virtual ~IPipelineVK();

			auto getVkPipeline() noexcept -> VkPipeline*;

		private:
			ILogicalDeviceVK* logicalDevice;
			VkPipeline graphicsPipeline;
			PipelineDesc desc;
			auto createVkPipeline() noexcept -> void;
			auto createVkComputePipeline(ComputePipelineDesc const& desc) noexcept -> void;
		};
	}
}
