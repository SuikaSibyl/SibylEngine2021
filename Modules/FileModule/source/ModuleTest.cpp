#include "ModuleTest.h"

#include <Core/module.h>

namespace SIByL
{
	namespace File
	{
		void ModuleTest::Test()
		{
			SIByL::Core::SLog::Core_Trace(0, "HELLO");
			//S_CORE_TEST(true, "True Test", "Failed because of Failure");
			//S_CORE_TEST(false, "Failed Test", "Failed because of Failure");
		}

	}
}