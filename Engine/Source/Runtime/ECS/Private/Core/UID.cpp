module;
#include <cstdint>
#include <stdlib.h> 
#include <time.h> 
#include <thread>
#include <random>
module ECS.UID;
import Network.Context;

namespace SIByL::ECS
{
	std::default_random_engine e;
	std::uniform_int_distribution<uint64_t> u(0, -1);

	auto UniqueID::RequestUniqueID() noexcept -> UID
	{
		// basically 19 numbers could be stored
		uint64_t id = 0;

		time_t now = time(0);
		tm ltm;
		localtime_s(&ltm, &now);

		//																		   Bits  AccumulatedLength 
		constexpr uint64_t year_mask = ((uint64_t)1 << 10) - 1;				// <<  0        10          10
		constexpr uint64_t month_mask = ((uint64_t)(1 << 4) - 1);				// << 10         4          14
		constexpr uint64_t day_mask = (((uint64_t)1 << 5) - 1);				// << 14         5          19
		constexpr uint64_t hour_mask = (((uint64_t)1 << 5) - 1);				// << 19		 5          24
		constexpr uint64_t minute_mask = (((uint64_t)1 << 6) - 1);			// << 24		 6          30
		constexpr uint64_t second_mask = (((uint64_t)1 << 6) - 1);			// << 30		 6          36
		constexpr uint64_t ipv4_3_mask = (((uint64_t)1 << 8) - 1);			// << 36	     8          44
		constexpr uint64_t thread_mask = (((uint64_t)1 << 4) - 1);			// << 44		 4          48
		constexpr uint64_t random_mask = (((uint64_t)1 << 16) - 1);			// << 48		16          64

		id += (uint64_t(ltm.tm_year) & year_mask) << 0;
		id += (uint64_t(ltm.tm_mon) & month_mask)		 << 10;
		id += (uint64_t(ltm.tm_mday) & day_mask)         << 14;
		id += (uint64_t(ltm.tm_hour) & hour_mask)		 << 19;
		id += (uint64_t(ltm.tm_min) & minute_mask)		 << 24;
		id += (uint64_t(ltm.tm_sec) & second_mask)		 << 30;

		Network::IP ip = Network::Context::instance()->getLocalIP();
		id += (uint64_t(ip.seg[3]) & ipv4_3_mask)		 << 36;

		std::thread::id tid = std::this_thread::get_id();
		unsigned int nId = *(unsigned int*)((char*)&tid);
		id += (uint64_t(nId) & thread_mask) << 44;

		id += (u(e) & random_mask) << 48;

		return id;
	}
}