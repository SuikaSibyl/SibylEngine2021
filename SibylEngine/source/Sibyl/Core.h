#pragma once

#ifdef SIBYL_PLATFORM_WINDOWS
	#ifdef SIBYL_BUILD_DLL
		#define SIByL_API __declspec(dllexport)
	#else
		#define SIByL_API __declspec(dllimport)
	#endif
#else
	#error Sibyl Only Support Windows Now!
#endif // SIBYL_PLATFORM_WINDOWS
