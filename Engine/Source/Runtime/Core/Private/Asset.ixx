module;
#include <string>
#include <filesystem>
export module Core.Asset;
import Core.Buffer;

namespace SIByL
{
	inline namespace Core
	{
		export class Asset
		{
		public:
			auto getIdentifier() noexcept -> uint64_t { return identifier; }
			auto getAttachedPath() noexcept -> std::filesystem::path { return attachedFile; }

		protected:
			uint64_t identifier = 0;
			std::filesystem::path attachedFile;
		};
	}
}