#pragma once

#include "SIByLpch.h"

//#define SIBYL_DYNAMIC_LINK
#ifdef SIBYL_PLATFORM_WINDOWS
	#ifdef SIBYL_DYNAMIC_LINK
		#ifdef SIBYL_BUILD_DLL
			#define SIByL_API __declspec(dllexport)
		#else
			#define SIByL_API __declspec(dllimport)
		#endif
	#else
		#define SIByL_API
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

#define BIND_EVENT_FN(x) std::bind(&x, this, std::placeholders::_1)

namespace SIByL
{
	template<typename T>
	using Scope = std::unique_ptr<T>;
	template<typename T, typename ... Args>
	constexpr Scope<T> CreateScope(Args&& ... args)
	{
		return std::make_unique<T>(std::forward<Args>(args)...);
	}

	template<typename T>
	using Ref = std::shared_ptr<T>;
	template<typename T, typename ... Args>
	constexpr Ref<T> CreateRef(Args&& ... args)
	{
		return std::make_shared<T>(std::forward<Args>(args)...);
	}
}