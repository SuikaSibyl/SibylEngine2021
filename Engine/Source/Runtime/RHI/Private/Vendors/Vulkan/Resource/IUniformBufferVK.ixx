module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IUniformBuffer.VK;
import Core.Buffer;
import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;
import RHI.IBuffer.VK;
import RHI.IUniformBuffer;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IUniformBufferVK :public IUniformBuffer
		{
		public:
			IUniformBufferVK();
			IUniformBufferVK(uint32_t const& size, ILogicalDeviceVK* logical_device);
			virtual ~IUniformBufferVK() = default;

			virtual auto updateBuffer(Buffer* buffer) noexcept -> void override;
			virtual auto getSize() noexcept -> uint32_t override;
			auto getVkBuffer() noexcept -> VkBuffer*;

		private:
			IBufferVK buffer;
			ILogicalDeviceVK* logicalDevice;
		};
	}
}