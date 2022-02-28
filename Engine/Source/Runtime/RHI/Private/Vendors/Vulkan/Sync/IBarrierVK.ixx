module;
#include <vector>
#include <vulkan/vulkan.h>
export module RHI.IBarrier.VK;
import RHI.IBarrier;
namespace SIByL
{
	namespace RHI
	{
		using std::vector;

		// ╔════════════════════╗
		// ║     IBarrierVK     ║
		// ╚════════════════════╝
		// Vulkan version of IBarrier
		// It is not a resource, but a description
		// Provide infomation a Vulkan barrier command need

		export class IBarrierVK :public IBarrier
		{
		public:
			IBarrierVK(BarrierDesc const& desc);
			virtual ~IBarrierVK() = default;

			auto getSrcStageMask() noexcept -> VkPipelineStageFlags { return srcStageMask; }
			auto getDstStageMask() noexcept -> VkPipelineStageFlags { return dstStageMask; }
			auto getDependencyFlags() noexcept -> VkDependencyFlags { return dependencyFlags; }
			auto getMemoryBarrierCount() noexcept -> uint32_t { return memoryBarriers.size(); }
			auto getBufferMemoryBarrierCount() noexcept -> uint32_t { return bufferMemoryBarriers.size(); }
			auto getImageMemoryBarrierCount() noexcept -> uint32_t { return imageMemoryBarriers.size(); }
			auto getMemoryBarrierData() noexcept -> VkMemoryBarrier*;
			auto getBufferMemoryBarrierData() noexcept -> VkBufferMemoryBarrier*;
			auto getImageMemoryBarrierData() noexcept -> VkImageMemoryBarrier*;

		private:
			VkPipelineStageFlags srcStageMask;
			VkPipelineStageFlags dstStageMask;
			VkDependencyFlags dependencyFlags;
			vector<VkMemoryBarrier> memoryBarriers;
			vector<VkBufferMemoryBarrier> bufferMemoryBarriers;
			vector<VkImageMemoryBarrier> imageMemoryBarriers;
		};

	}
}