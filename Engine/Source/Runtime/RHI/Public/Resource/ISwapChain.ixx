module;

export module RHI.ISwapChain;
import RHI.IResource;

namespace SIByL
{
	namespace RHI
	{
		export class ISwapChain :public IResource
		{
		public:
			virtual ~ISwapChain() = default;
			
		};
	}
}
