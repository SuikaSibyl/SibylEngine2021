module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IVertexBuffer.VK;
import Core.Buffer;
import RHI.IResource;
import RHI.IEnum;
import RHI.IBuffer;
import RHI.IVertexBuffer;
import RHI.ILogicalDevice.VK;
import RHI.IBuffer.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IVertexBufferVK :public IVertexBuffer
		{
		public:
			IVertexBufferVK(Buffer* buffer, ILogicalDeviceVK* logical_device);
			virtual ~IVertexBufferVK();
			auto getVkBuffer() noexcept ->VkBuffer*;

		private:
			IBufferVK buffer;
			ILogicalDeviceVK* logicalDevice;
		};
	}
}