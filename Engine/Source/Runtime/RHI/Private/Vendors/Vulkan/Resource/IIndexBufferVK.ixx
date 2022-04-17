module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IIndexBuffer.VK;
import Core.Buffer;
import RHI.IIndexBuffer;
import RHI.IBuffer.VK;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		export class IIndexBufferVK :public IIndexBuffer
		{
		public:
			IIndexBufferVK(Buffer* buffer, uint32_t const& size, ILogicalDeviceVK* logical_device);
			virtual ~IIndexBufferVK() = default;

			virtual auto getElementSize() noexcept -> uint32_t override;
			virtual auto getIndicesCount() noexcept -> uint32_t override;

			auto getVkBuffer() noexcept ->VkBuffer*;
			auto getVkIndexType() noexcept -> VkIndexType;

		private:
			uint32_t elementSize;
			uint32_t indicesCount;
			IBufferVK buffer;
			ILogicalDeviceVK* logicalDevice;
		};
	}
}