#pragma once

#include "SIByLpch.h"

//#define SIByL_DX12_CORE
#define SIByL_OpenGL_CORE

#ifdef SIBYL_PLATFORM_WINDOWS
	#ifdef SIBYL_BUILD_DLL
		#define SIByL_API __declspec(dllexport)
	#else
		#define SIByL_API __declspec(dllimport)
	#endif
#else
	#error Sibyl Only Support Windows Now!
#endif // SIBYL_PLATFORM_WINDOWS

#ifdef SIBYL_ENABLE_ASSETS
	#define SIByL_ASSERT(x, ...) {if(!(x)){SIByL_CLIENT_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak();}}
	#define SIByL_CORE_ASSERT(x, ...) {if(!(x)){SIByL_CORE_ERROR("Assertion Failed: {0}", __VA_ARGS__); __debugbreak();}}
#else
	#define SIByL_ASSERT(x, ...)
	#define SIByL_CORE_ASSERT(x, ...)
#endif
#define BIT(x) (1 << x)