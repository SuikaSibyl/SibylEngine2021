module;
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
module RHI.IBuffer;
import RHI.IResource;
import RHI.IEnum;

namespace SIByL::RHI
{
	BufferLayout::BufferLayout(std::initializer_list<BufferElement> const& _elements)
		:elements(_elements)
	{
		// Calculate Offsets and Stride
		uint32_t offset = 0;
		stride = 0;
		for (auto& element : elements)
		{
			element.offset = offset;
			offset += sizeofDataType(element.type);
			stride += sizeofDataType(element.type);
		}
	}

	auto BufferLayout::getElements() noexcept -> std::vector<BufferElement>&
	{
		return elements;
	}

	auto BufferLayout::getStride() const noexcept -> uint32_t
	{
		return stride;
	}

	auto BufferLayout::begin() noexcept -> BufferLayout::iter
	{
		return elements.begin();
	}

	auto BufferLayout::end() noexcept -> BufferLayout::iter
	{
		return elements.end();
	}
}
