module;
#include <slang.h>
#include <slang-com-ptr.h>
#include <string_view>
export module RHI.ICompileSession;
import Core.SObject;

using Slang::ComPtr;

namespace SIByL
{
	namespace RHI::SLANG
	{
		export class ICompileSession :public SObject
		{
		public:
			ICompileSession();

			auto loadModule(std::string_view module_name, std::string_view entry_point) noexcept -> bool;

		private:
			ComPtr<slang::ISession> session;
		};
	}
}