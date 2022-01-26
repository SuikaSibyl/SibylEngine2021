module;

export module RHI.ISwapChain;
import Core.SObject;

namespace SIByL
{
	namespace RHI
	{
		export class ISwapChain :public SObject
		{
		public:
			virtual ~ISwapChain() = default;
			
		};
	}
}
