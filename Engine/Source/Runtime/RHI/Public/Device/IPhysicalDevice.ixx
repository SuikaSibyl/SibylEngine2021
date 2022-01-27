export module RHI.IPhysicalDevice;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		export class IPhysicalDevice :public SObject
		{
		public:
			virtual ~IPhysicalDevice() = default;

			virtual auto isDebugLayerEnabled() noexcept -> bool { return true; }
		};
	}
}
