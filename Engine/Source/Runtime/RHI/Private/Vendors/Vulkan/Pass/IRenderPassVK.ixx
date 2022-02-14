module;
#include <vulkan/vulkan.h>
export module RHI.IRenderPass.VK;
import RHI.IRenderPass;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IRenderPassVK :public IRenderPass
		{
		public:
			IRenderPassVK() = default;
			IRenderPassVK(IRenderPassVK&&) = default;
			virtual ~IRenderPassVK();

			auto createRenderPass() noexcept -> void;
			auto getRenderPass() noexcept -> VkRenderPass*;

		private:
			ILogicalDeviceVK* logicalDevice;
			VkRenderPass renderPass;
		};
	}
}
