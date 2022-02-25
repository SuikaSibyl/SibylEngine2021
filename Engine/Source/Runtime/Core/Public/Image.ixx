module;
#include <cstdint>
#include <filesystem>
export module Core.Image;
import Core.Buffer;

namespace SIByL
{
	inline namespace Core
	{
		export class Image
		{
		public:
			Image(std::filesystem::path path);
			virtual ~Image();

			auto getWidth() noexcept -> uint32_t;
			auto getHeight() noexcept -> uint32_t;
			auto getChannels() noexcept -> uint32_t;
			auto getSize() noexcept -> uint32_t;
			auto getData() noexcept -> char*;

		private:
			uint32_t width;
			uint32_t height;
			uint32_t channels;
			char* data;
		};
	}
}