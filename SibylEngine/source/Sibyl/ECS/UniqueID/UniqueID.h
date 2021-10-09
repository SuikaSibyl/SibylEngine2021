#pragma once

#include <ctime>

namespace SIByL
{
	class UniqueID
	{
	public:
		static uint64_t RequestUniqueID();
		static void InsertUidEidPair(uint64_t uid, entt::entity eid);
		static entt::entity GetEid(uint64_t uid);

	private:
		static std::unordered_map<uint64_t, entt::entity> Uid2EidMap;
	};
}