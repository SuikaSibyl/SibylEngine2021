module;
#include <vulkan/vulkan.h>
module RHI.IPhysicalDevice.VK;

namespace SIByL::RHI
{
#ifdef _DEBUG
	const bool enableValidationLayers = true;
#else
	const bool enableValidationLayers = false;
#endif
	
	auto IPhysicalDeviceVK::initialize() -> bool
	{
		enableDebugLayer = enableValidationLayers;

		return true;
	}

	auto IPhysicalDeviceVK::isDebugLayerEnabled() noexcept -> bool
	{
		return enableDebugLayer;
	}
}