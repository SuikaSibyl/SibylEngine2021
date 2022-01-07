module;
#include <vector>
#include <functional>
#include <string_view>
export module Core.Test;

export auto SE_TEST_CHECK(bool x, std::string_view desc) noexcept -> void;
export auto SE_TEST_EXEC_ALL() noexcept -> void;

namespace SIByL::Core
{
	using TestFn = std::function<void()>;
	struct TestPool;

	export struct Testcase
	{
		Testcase(TestFn fn, std::string_view desc);

		TestFn fn;
		std::string_view desc;

		friend struct TestPool;
	};

	struct TestPool
	{
		void execAll();
		void addTestcase(Testcase* testcase);

		std::vector<Testcase*> tests;
		uint64_t pass_number = 0;
		uint64_t fail_number = 0;
		uint64_t fail_number_local = 0;
		bool is_last_testcase = false;
	};
}

export using SE_TEST_CASE = SIByL::Core::Testcase;