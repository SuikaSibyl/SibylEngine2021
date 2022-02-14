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
			IPipelineVK() = default;
			IPipelineVK(IPipelineVK&&) = default;
			virtual ~IPipelineVK();

			auto getVkPipeline() noexcept -> VkPipeline*;

		private:
			ILogicalDeviceVK* logicalDevice;
			VkPipeline graphicsPipeline;
			auto createVkPipeline() noexcept -> void;
		};
	}
}
