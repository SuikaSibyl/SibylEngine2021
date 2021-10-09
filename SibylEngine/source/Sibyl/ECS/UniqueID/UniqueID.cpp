#include "SIByLpch.h"
#include "UniqueID.h"

#include "NetworkModule/include/NetworkModule.h"
#include <stdlib.h> 
#include <time.h> 
#include<thread>

using namespace std;

namespace SIByL
{
	std::unordered_map<uint64_t, entt::entity> UniqueID::Uid2EidMap;

	uint64_t UniqueID::RequestUniqueID()
	{
		// basically 19 numbers could be stored
		uint64_t id = 0;

		time_t now = time(0);
		tm* ltm = localtime(&now);

		id += (ltm->tm_year-100) * 10000000000000000;		// Year		3	03
		id += uint64_t(1 + ltm->tm_mon) * 100000000000000;	// Month	2	05
		id += uint64_t(ltm->tm_mday) * 1000000000000;		// Day		2	07
		id += uint64_t(ltm->tm_hour) * 10000000000;			// Hour		2	09
		id += uint64_t(ltm->tm_min) * 100000000;			// Minute	2	11
		id += uint64_t(ltm->tm_sec) * 1000000;				// Second	2	13

		SIByLNetwork::IP ip = SIByLNetwork::NetworkContext::GetLocalIP();
		id += uint64_t(ip.seg[3] % 100) * 10000;			// IPV4Seg4	2	15

		std::thread::id tid = std::this_thread::get_id();
		unsigned int nId = *(unsigned int*)((char*)&tid);
		id += (nId % 10000);								// IPV4Seg4	4	19

		return id;
	}

	void UniqueID::InsertUidEidPair(uint64_t uid, entt::entity eid)
	{
		Uid2EidMap[uid] = eid;
	}
	
	entt::entity UniqueID::GetEid(uint64_t uid)
	{
		return Uid2EidMap[uid];
	}

}