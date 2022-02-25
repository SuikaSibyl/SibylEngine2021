module;
#include <cstdint>
#include <vulkan/vulkan.h>
module RHI.IFence.VK;
import RHI.IFence;
import Core.Log;
import Core.SObject;
import RHI.ILogicalDevice.VK;

namespace SIByL
{
	namespace RHI
	{
		auto createVKFence(VkDevice* logical_device, VkFence* fence) noexcept -> void
		{
			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			if(vkCreateFence(*logical_device, &fenceInfo, nullptr, fence) != VK_SUCCESS)
			{
				SE_CORE_ERROR("VULKAN :: failed to create fence");
			}
		}

		IFenceVK::IFenceVK(ILogicalDeviceVK* logical_device)
			:logicalDevice(logical_device)
		{
			createVKFence(&(logicalDevice->getDeviceHandle()), &vkFence);
		}
		
		auto IFenceVK::wait() noexcept -> void
		{
			vkWaitForFences(logicalDevice->getDeviceHandle(), 1, &vkFence, VK_TRUE, UINT64_MAX);
		}

		auto IFenceVK::reset() noexcept -> void
		{
			vkResetFences(logicalDevice->getDeviceHandle(), 1, &vkFence);
		}

		auto IFenceVK::getVkFence() noexcept -> VkFence*
		{
			return &vkFence;
		}

		IFenceVK::~IFenceVK()
		{
			if (vkFence)
				vkDestroyFence(logicalDevice->getDeviceHandle(), vkFence, nullptr);
		}
	}
}
