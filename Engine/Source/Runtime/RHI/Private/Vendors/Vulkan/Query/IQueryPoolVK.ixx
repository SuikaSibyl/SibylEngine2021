module;
#include <vulkan/vulkan.h>
export module RHI.IQueryPool.VK;
import RHI.IQueryPool;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;

namespace SIByL::RHI
{
	export struct IQueryPoolVK :public IQueryPool
	{
		IQueryPoolVK(QueryPoolDesc const& desc, ILogicalDeviceVK* logical_device);
		~IQueryPoolVK();

		auto getQueryPool() noexcept -> VkQueryPool*;

		virtual auto reset(uint32_t const& start, uint32_t const& size) noexcept -> void override;
		virtual auto fetchResult(uint32_t const& start, uint32_t const& size, uint64_t* result) noexcept -> bool override;
	private:
		VkQueryPool queryPool;
		ILogicalDeviceVK* logicalDevice;
	};
}