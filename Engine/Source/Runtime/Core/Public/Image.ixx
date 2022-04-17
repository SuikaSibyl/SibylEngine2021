module;
#include <cstdint>
#include <filesystem>
export module Core.Image;
import Core.Buffer;
import Core.BitFlag;
import Core.LinearAlgebra;

namespace SIByL
{
	inline namespace Core
	{
		export class Image
		{
		public:
			Image() = default;
			Image(Image&& image);
			Image(std::filesystem::path path);
			Image(uint32_t width, uint32_t height);
			Image& operator=(Image&& src);
			virtual ~Image();

			auto getWidth() noexcept -> uint32_t;
			auto getHeight() noexcept -> uint32_t;
			auto getChannels() noexcept -> uint32_t;
			auto getSize() noexcept -> uint32_t;
			auto getData() noexcept -> char*;

			void setPixel(uint32_t x, uint32_t y, const Vec4f& color);
			void savePPM(std::filesystem::path path) const;
			void saveTGA(std::filesystem::path path) const;

			enum struct ImageAttribute :uint32_t
			{
				FROM_STB = 1 << 0,
			};
			uint32_t attributes = 0;
			uint32_t width;
			uint32_t height;
			uint32_t channels;
			uint32_t size;

			char* data = nullptr;
		};
	}
}