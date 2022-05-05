module;
#include <vulkan/vulkan.h>
module RHI.IQueryPool.VK;
import Core.Log;
import RHI.IQueryPool;
import RHI.ILogicalDevice;
import RHI.ILogicalDevice.VK;
 
namespace SIByL::RHI
{
    auto QueryType2VkQueryType(QueryType type) noexcept -> VkQueryType
    {
        switch (type)
        {
        case QueryType::OCCLUSION:
            return VK_QUERY_TYPE_OCCLUSION;
        case QueryType::PIPELINE_STATISTICS:
            return VK_QUERY_TYPE_PIPELINE_STATISTICS;
        case QueryType::TIMESTAMP:
            return VK_QUERY_TYPE_TIMESTAMP;
        default:
            break;
        }
        return VK_QUERY_TYPE_OCCLUSION;
    }

    auto createQueryPool(QueryPoolDesc const& desc, ILogicalDeviceVK* logical_device, VkQueryPool* queryPool) noexcept -> void
    {
        VkQueryPoolCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        createInfo.pNext = nullptr; // Optional
        createInfo.flags = 0; // Reserved for future use, must be 0!

        createInfo.queryType = QueryType2VkQueryType(desc.type);
        createInfo.queryCount = desc.number; // REVIEW

        VkResult result = vkCreateQueryPool(logical_device->getDeviceHandle(), &createInfo, nullptr, queryPool);
        if (result != VK_SUCCESS)
        {
            SE_CORE_ERROR("RHI :: VULKAN :: Failed to create time query pool!");
        }
    }

    IQueryPoolVK::IQueryPoolVK(QueryPoolDesc const& desc, ILogicalDeviceVK* logical_device)
        :logicalDevice(logical_device)
	{
        createQueryPool(desc, logical_device, &queryPool);
	}

    auto IQueryPoolVK::getQueryPool() noexcept -> VkQueryPool*
    {
        return &queryPool;
    }

    IQueryPoolVK::~IQueryPoolVK()
    {
        vkDestroyQueryPool(logicalDevice->getDeviceHandle(), queryPool, nullptr);
    }

    auto IQueryPoolVK::reset(uint32_t const& start, uint32_t const& size) noexcept -> void
    {
        vkResetQueryPool(logicalDevice->getDeviceHandle(), queryPool, start, 0);
    }

    auto IQueryPoolVK::fetchResult(uint32_t const& start, uint32_t const& size, uint64_t* result) noexcept -> bool
    {
        VkResult vkres = vkGetQueryPoolResults(
            logicalDevice->getDeviceHandle(), 
            queryPool, 
            start, 
            size,
            sizeof(uint64_t) * size, 
            result, 
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);

        if (vkres == VK_NOT_READY)
        {
            return false;
        }
        else if (vkres == VK_SUCCESS)
        {
            return true;
        }
        else
        {
            SE_CORE_ERROR("RHI :: RHI :: IQueryPoolVK::fetchResult() Failed to receive query results!");
        }
    }
}