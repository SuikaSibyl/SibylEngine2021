module;
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IRenderPass.VK;
import Core.Color;
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
			auto getVkClearValues() noexcept -> VkClearValue* { return clearValues.data(); }
			auto getVkClearValueSize() noexcept -> uint32_t { return clearValues.size(); }

		private:
			std::vector<VkClearValue> clearValues;
			ILogicalDeviceVK* logicalDevice;
			VkRenderPass renderPass;
		};
	}
}
