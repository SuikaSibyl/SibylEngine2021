module;
#include <unordered_map>
export module RHI.IDeviceGlobal;
import Core.MemoryManager;
import RHI.ILogicalDevice;
import RHI.ICommandPool;
import RHI.IFactory;

namespace SIByL
{
	namespace RHI
	{
		export struct PerDeviceGlobal
		{
			PerDeviceGlobal() = default;
			PerDeviceGlobal(ILogicalDevice* device);

			auto getTransientCommandPool() noexcept -> ICommandPool*;
			auto getResourceFactory() noexcept -> IResourceFactory*;

			MemScope<IResourceFactory> resourceFactory;
			MemScope<ICommandPool> transientCommandPool;
		};

		export class DeviceToGlobal
		{
		public:
			static auto getGlobal(ILogicalDevice* device) -> PerDeviceGlobal*;
			static auto releaseGlobal() -> void;
			static auto removeDevice(ILogicalDevice* device)->void;

		private:
			static DeviceToGlobal deviceGlobal;
			std::unordered_map<ILogicalDevice*, PerDeviceGlobal> map;
		};
	}
}