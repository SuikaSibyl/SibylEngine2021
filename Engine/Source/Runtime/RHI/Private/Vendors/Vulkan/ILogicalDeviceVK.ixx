module;
#include <vulkan/vulkan.h>
export module RHI.LogicalDevice.VK;
import RHI.GraphicContext.VK;

namespace SIByL
{
	namespace RHI
	{
		export class ILogicalDeviceVK
		{
		public:
			auto createLogicalDevice(IGraphicContextVK& context) -> void;

		private:
			VkDevice device;
		};
	}
}