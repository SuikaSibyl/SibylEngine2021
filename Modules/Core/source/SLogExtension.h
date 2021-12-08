#pragma once

#include "SLog.h"

// Assert
#define S_CORE_ASSERT(x, ...) {if(!(x)){S_CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak();}}
#define S_ASSERT(x, ...) {if(!(x)){S_CLIENT_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak();}}

// Test
#define S_CORE_TEST(x, ...) {\
	if(!(x)){S_CORE_ERROR("\t[X] Failed | Test {0}:\t {1}", __VA_ARGS__);}\
	else{S_CORE_TRACE("\t[O] Passed | Test {0}", __VA_ARGS__);}}

#define S_TEST(x, ...) {\
	if(!(x)){S_CLIENT_ERROR("\t[X] Failed | Test {0}:\t {1}", __VA_ARGS__);}\
	else{S_CLIENT_TRACE("\t[O] Passed | Test {0}", __VA_ARGS__);}}
