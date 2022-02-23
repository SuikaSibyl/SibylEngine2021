module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
export module RHI.IBuffer;
import RHI.IResource;
import RHI.IEnum;

namespace SIByL
{
	namespace RHI
	{
		export struct BufferElement
		{
			DataType type;
			std::string_view name;
			uint32_t offset;
		};

		export class BufferLayout
		{
		public:
			BufferLayout(std::initializer_list<BufferElement> const& _elements);
			using iter = std::vector<BufferElement>::iterator;
			auto getElements() noexcept -> std::vector<BufferElement>&;
			auto getStride() const noexcept -> uint32_t;
			auto begin() noexcept -> iter;
			auto end() noexcept -> iter;

		private:
			std::vector<BufferElement> elements;
			uint32_t stride = 0;
		};

		// A Buffer is an array of data, such as a mesh's positional data, color data, 
		// index data, etc. Similar rules for images apply to buffers in Vulkan and WebGPU.
		// ╭──────────────┬──────────────────────────────────╮
		// │  Vulkan	  │   vk::Buffer & vk::BufferView    │
		// │  DirectX 12  │   ID3D12Resource                 │
		// │  OpenGL      │   Varies by OS                   │
		// ╰──────────────┴──────────────────────────────────╯

		export class IBuffer :public IResource
		{
		public:
			IBuffer() = default;
			IBuffer(IBuffer&&) = default;
			virtual ~IBuffer() = default;


		};
	}
}
