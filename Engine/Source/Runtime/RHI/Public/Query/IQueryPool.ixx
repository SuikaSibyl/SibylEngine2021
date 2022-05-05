module;
#include <cstdint>
export module RHI.IQueryPool;

namespace SIByL::RHI
{
	export enum struct QueryType
	{
		OCCLUSION = 0,
		PIPELINE_STATISTICS = 1,
		TIMESTAMP = 2,
	};

	export struct QueryPoolDesc
	{
		QueryType type;
		uint32_t number;
	};

	export struct IQueryPool
	{
		virtual ~IQueryPool() = default;

		virtual auto reset(uint32_t const& start, uint32_t const& size) noexcept -> void = 0;
		virtual auto fetchResult(uint32_t const& start, uint32_t const& size, uint64_t* result) noexcept -> bool = 0;
	};
}