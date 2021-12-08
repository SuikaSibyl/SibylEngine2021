#pragma once

#include "../vendor/slang/include/slang.h"
#include "../vendor/slang/include/slang-com-ptr.h"

namespace SIByL
{
	namespace ShaderModule
	{
		using Slang::ComPtr;

		class SlangMachine
		{
		public:
			void CreateSession()noexcept;


		protected:

			static auto GetGlobalSession()noexcept->ComPtr<slang::IGlobalSession>;
			static auto ReleaseGlobalSession()noexcept->void;

		protected:
			ComPtr<slang::ISession> mSession = nullptr;

			static ComPtr<slang::IGlobalSession> sGlobalSession;
		};
	}
}