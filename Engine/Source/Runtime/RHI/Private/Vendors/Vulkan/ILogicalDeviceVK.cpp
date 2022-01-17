module;
#include <vulkan/vulkan.h>
#include <vector>
module RHI.ILogicalDevice.VK;
import Core.Log;
import RHI.IPhysicalDevice.VK;

namespace SIByL::RHI
{
	ILogicalDeviceVK::ILogicalDeviceVK(IPhysicalDeviceVK* physicalDevice)
		:physicalDevice(physicalDevice)
	{

	}

	auto ILogicalDeviceVK::initialize() -> bool
	{
		return true;
	}

	auto ILogicalDeviceVK::destroy() -> bool
	{
		vkDestroyDevice(device, nullptr);
		return true;
	}

}
