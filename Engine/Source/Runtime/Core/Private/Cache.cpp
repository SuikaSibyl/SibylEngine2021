module;
#include <filesystem>
module Core.Cache;
import Core.File;

namespace SIByL::Core
{
	CacheBrain::CacheBrain()
	{
		sourceLoader.addSearchPath("./assets");
		cacheLoader.addSearchPath("./cache");
	}

	auto CacheBrain::instance() noexcept -> CacheBrain*
	{
		static CacheBrain cache_brain;
		return &cache_brain;
	}

}