module;
#include <iostream>
#include <string_view>
module Core.Test;
import Core.Log;

namespace SIByL::Core
{
	TestPool g_test_pool;

	Testcase::Testcase(TestFn fn, std::string_view desc)
		:fn(fn), desc(desc)
	{
		g_test_pool.addTestcase(this);
	}

	void TestPool::addTestcase(Testcase* testcase)
	{
		tests.push_back(testcase);
	}

	void TestPool::execAll()
	{
		::SE_CORE_DEBUG("¨q BEGIN RUN ALL TESTCASES");
		for (auto iter = tests.begin(); iter != tests.end(); iter++)
		{
			if (iter + 1 == tests.end())
			{
				is_last_testcase = true;
			}
			else
			{
				is_last_testcase = false;
			}
			::SE_CORE_TRACE("©¦ ¨q©¤ TESTCASE: {0}", (*iter)->desc);
			fail_number_local = 0;
			(*iter)->fn();
			::SE_CORE_TRACE("©¦ ¨t©¤ CHECK FAILURE : {0}", fail_number_local);
		}
		::SE_CORE_DEBUG("¨t FINISH ALL TESTCASES, PASS: {0}, FAIL: {1}", pass_number, fail_number);
	}

	auto testCeck(bool x, std::string_view desc) noexcept -> void
	{
		static std::string_view prefix  = "©¦ ©¦    ";

		if (x) {
			g_test_pool.pass_number++;
			::SE_CORE_TRACE("{0}CHECK PASS: {1}", prefix, desc);
		} else {
			g_test_pool.fail_number++;
			g_test_pool.fail_number_local++;
			::SE_CORE_ERROR("{0}CHECK FAIL: {1}", prefix, desc);
		}
	}
}

auto SE_TEST_CHECK(bool x, std::string_view desc) noexcept -> void
{
	SIByL::Core::testCeck(x, desc);
}

auto SE_TEST_EXEC_ALL() noexcept -> void
{
	SIByL::Core::g_test_pool.execAll();
}
