module;

export module RHI.IBuffer;
import RHI.IResource;

namespace SIByL
{
	namespace RHI
	{
		export class IBuffer :public IResource
		{
		public:
			IBuffer() = default;
			IBuffer(IBuffer&&) = default;
			virtual ~IBuffer() = default;


		};
	}
}
