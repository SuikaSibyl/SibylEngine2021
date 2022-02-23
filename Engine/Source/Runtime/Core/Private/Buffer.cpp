module;
#include <memory>
#include <cstdint>
module Core.Buffer;
import Core.MemoryManager;

namespace SIByL
{
	inline namespace Core
	{
		Buffer::Buffer()
			: data(nullptr)
			, size(0)
			, alignment(alignof(uint32_t))
		{}

		Buffer::Buffer(size_t const& _size, size_t const& _alignment)
			: size(_size)
			, alignment(_alignment)
		{
			data = (char*)Memory::instance()->allocate(size);
		}

		Buffer::Buffer(void* outer_buffer, size_t const& _size, size_t const& _alignment)
			: size(_size)
			, alignment(_alignment)
			, isProxy(true)
			, data((char*)outer_buffer)
		{}

		Buffer::Buffer(Buffer const& rhs)
		{
			release();

			size = rhs.size;
			alignment = rhs.alignment;
			data = (char*)Memory::instance()->allocate(size);
			isProxy = rhs.isProxy;
			memcpy(data, rhs.data, size);
		}

		Buffer::Buffer(Buffer&& rhs)
		{
			release();

			size = rhs.size;
			alignment = rhs.alignment;
			data = rhs.data;
			isProxy = rhs.isProxy;

			rhs.size = 0;
			rhs.alignment = 4;
			rhs.data = nullptr;
		}

		Buffer::~Buffer()
		{
			release();
		}

		void Buffer::release()
		{
			if (data && !isProxy) Memory::instance()->free(data, size);
		}

		Buffer& Buffer::operator = (Buffer const& rhs)
		{
			release();

			if (data) Memory::instance()->free(data, size);
			size = rhs.size;
			alignment = rhs.alignment;
			data = (char*)Memory::instance()->allocate(size);
			isProxy = rhs.isProxy;
			memcpy(data, rhs.data, size);
			return *this;
		}

		Buffer& Buffer::operator = (Buffer&& rhs)
		{
			release();

			if (data) Memory::instance()->free(data, size);
			data = rhs.data;
			size = rhs.size;
			alignment = rhs.alignment;
			isProxy = rhs.isProxy;
			rhs.size = 0;
			rhs.alignment = 4;
			rhs.data = nullptr;
			return *this;
		}

		auto Buffer::getData() const -> char*
		{
			return data;
		}

		auto Buffer::getpSize()->size_t*
		{
			return &size;
		}

		auto Buffer::getSize() const ->size_t
		{
			return size;
		}
	}
}