module;
#include <cstdint>
export module ECS.UID;

namespace SIByL::ECS
{
	export using UID = uint64_t;

	export class UniqueID
	{
	public:
		static auto RequestUniqueID() noexcept -> UID;
	};
}