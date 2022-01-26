module;

export module RHI.IResource;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		export class IResource :public SObject
		{
		public:
			IResource() = default;
			IResource(IResource&&) = default;
			IResource(IResource const&) = delete;
			virtual ~IResource() = default;

		};
	}
}
