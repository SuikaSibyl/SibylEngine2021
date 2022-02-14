module;
#include <slang.h>
#include <slang-com-ptr.h>
export module RHI.IGlobalSession;
import Core.SObject;

using Slang::ComPtr;

namespace SIByL
{
	namespace RHI::SLANG
	{
		export class IGlobalSession :public SObject
		{
		public:
			IGlobalSession();
			static IGlobalSession* instance();
			slang::IGlobalSession* getGlobalSession();

		private:
			ComPtr<slang::IGlobalSession> slangGlobalSession;
		};
	}
}