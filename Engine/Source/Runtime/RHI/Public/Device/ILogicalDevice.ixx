module;

export module RHI.ILogicalDevice;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		export class ILogicalDevice :public SObject
		{
		public:
			virtual ~ILogicalDevice() = default;

		};
	}
}
