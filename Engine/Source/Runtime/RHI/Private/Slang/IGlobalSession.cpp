module;
#include <slang.h>
#include <slang-com-ptr.h>
module RHI.IGlobalSession;
import Core.SObject;

using Slang::ComPtr;

namespace SIByL
{
	namespace RHI::SLANG
	{
		IGlobalSession::IGlobalSession()
		{
			SlangResult ret = slang::createGlobalSession(slangGlobalSession.writeRef());
		}

		IGlobalSession* IGlobalSession::instance()
		{
			static IGlobalSession s_global_session;
			return &s_global_session;
		}

		slang::IGlobalSession* IGlobalSession::getGlobalSession()
		{
			return slangGlobalSession.get();
		}
	}
}