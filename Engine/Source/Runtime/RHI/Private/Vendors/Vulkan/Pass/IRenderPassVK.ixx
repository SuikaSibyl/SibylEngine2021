module;
#include <vulkan/vulkan.h>
export module RHI.IRenderPass.VK;
import RHI.IRenderPass;
import RHI.ILogicalDevice.VK;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		export class IRenderPassVK :public IRenderPass
		{
		public:
			IRenderPassVK(RenderPassDesc const& desc, ILogicalDeviceVK* logical_device);
			IRenderPassVK(IRenderPassVK&&) = default;
			virtual ~IRenderPassVK();

			auto createRenderPass(RenderPassDesc const& desc) noexcept -> void;
			auto getRenderPass() noexcept -> VkRenderPass*;

		private:
			ILogicalDeviceVK* logicalDevice;
			VkRenderPass renderPass;
		};
	}
}
