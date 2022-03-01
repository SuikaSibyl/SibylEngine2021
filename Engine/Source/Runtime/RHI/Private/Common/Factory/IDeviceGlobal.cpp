module;
#include <unordered_map>
module RHI.IDeviceGlobal;
import Core.MemoryManager;
import RHI.ILogicalDevice;
import RHI.ICommandPool;
import RHI.IFactory;

namespace SIByL::RHI
{
	PerDeviceGlobal::PerDeviceGlobal(ILogicalDevice* device)
	{
		resourceFactory = MemNew<IResourceFactory>(device);

		transientCommandPool = resourceFactory->createCommandPool(CommandPoolDesc{
			QueueType::GRAPHICS,
			(uint32_t)CommandPoolAttributeFlagBits::RESET | (uint32_t)CommandPoolAttributeFlagBits::TRANSIENT
			});
	}

	auto PerDeviceGlobal::getTransientCommandPool() noexcept -> ICommandPool*
	{
		return transientCommandPool.get();
	}

	auto PerDeviceGlobal::getResourceFactory() noexcept -> IResourceFactory*
	{
		return resourceFactory.get();
	}

	DeviceToGlobal DeviceToGlobal::deviceGlobal;

	auto DeviceToGlobal::getGlobal(ILogicalDevice* device)->PerDeviceGlobal*
	{
		bool find = false;
		for (auto iter = deviceGlobal.map.begin(); iter != deviceGlobal.map.end(); iter++)
		{
			if (iter->first == device)
				find = true;
		}
		if (!find)
		{
			deviceGlobal.map.insert({ device, std::move(PerDeviceGlobal(device)) });
		}
		return &deviceGlobal.map[device];
	}

	auto DeviceToGlobal::removeDevice(ILogicalDevice* device)->void
	{
		auto iter = deviceGlobal.map.begin();
		for (; iter != deviceGlobal.map.end(); iter++)
		{
			if (iter->first == device)
			{
				deviceGlobal.map.erase(iter);
				return;
			}
		}
	}

	auto DeviceToGlobal::releaseGlobal() -> void
	{
		deviceGlobal.map.clear();
	}
}