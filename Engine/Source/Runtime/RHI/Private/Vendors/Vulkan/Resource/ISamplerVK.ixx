module;
#include <vulkan/vulkan.h>
export module RHI.ISampler.VK;
import RHI.ISampler;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class ISamplerVK : public ISampler
		{
		public:
			ISamplerVK(SamplerDesc const& desc, ILogicalDeviceVK* logical_device);
			virtual ~ISamplerVK();

			auto getVkSampler() noexcept -> VkSampler* { return &textureSampler; }

		private:
			VkSampler textureSampler;
			ILogicalDeviceVK* logicalDevice;
		};
	}
}