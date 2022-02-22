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

			virtual IResource& operator= (IResource&& rho);

        protected:
            ResourceType type;
            ResourceState state;
            ResourceFormat format;
		};

		IResource& IResource::operator= (IResource&& rho)
		{
			type = rho.type;
			state = rho.state;
			format = rho.format;
			return *this;
		}
	}
}
