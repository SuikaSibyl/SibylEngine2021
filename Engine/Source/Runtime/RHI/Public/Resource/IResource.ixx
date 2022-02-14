module;
#include <stdint.h>
export module RHI.IResource;
import Core.SObject;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		export class IResource :public SObject
		{
		public:
			IResource() = default;
			IResource(IResource&&) = default;
			virtual ~IResource() = default;

        protected:
            ResourceType type;
            ResourceState state;
            ResourceFormat format;
		};
	}
}
