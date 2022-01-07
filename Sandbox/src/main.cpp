#include <iostream>
#include <memory>

import Core.Assert;
import Core.Test;

int main()
{
	SE_TEST_EXEC_ALL();
}

static SE_TEST_CASE main_test_1([]() {
	SE_TEST_CHECK(1 == 1, "1 == 1");
	SE_TEST_CHECK(1 == 2, "1 == 2");
	return;
	}, "Hell0");

static SE_TEST_CASE main_test_2([]() {
	SE_TEST_CHECK(1 == 1, "1 == 1");
	SE_TEST_CHECK(2 == 2, "2 == 2");
	return; 
	}, "Hell1");