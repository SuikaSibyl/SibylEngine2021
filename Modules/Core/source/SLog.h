#pragma once
#include <iostream>

namespace SIByL
{
	namespace Core
	{
		class SLog
		{
		public:			
			static void Core_Trace(int i, ...);


		};
	}
}

#define S_CORE_TRACE(...)	::SIByL::Core::SLog::Core_Trace(__VA_ARGS__)
//#define S_CORE_INFO(...)	::SIByL::Core::SLog::GetCoreLogger()->info(__VA_ARGS__)
//#define S_CORE_WARN(...)	::SIByL::Core::SLog::GetCoreLogger()->warn(__VA_ARGS__)
//#define S_CORE_ERROR(...)	::SIByL::Core::SLog::GetCoreLogger()->error(__VA_ARGS__)
//
//#define S_CLIENT_TRACE(...)	::SIByL::Core::SLog::GetClientLogger()->trace(__VA_ARGS__)
//#define S_CLIENT_INFO(...)		::SIByL::Core::SLog::GetClientLogger()->info(__VA_ARGS__)
//#define S_CLIENT_WARN(...)		::SIByL::Core::SLog::GetClientLogger()->warn(__VA_ARGS__)
//#define S_CLIENT_ERROR(...)	::SIByL::Core::SLog::GetClientLogger()->error(__VA_ARGS__)
