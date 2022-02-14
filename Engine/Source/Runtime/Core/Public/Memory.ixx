module;

export module Core.MemoryManager;
import Core.Allocator;

namespace SIByL
{
	inline namespace Core
	{
		export class Memory
		{
		public:
			Memory();
			virtual ~Memory();

			static auto instance() noexcept -> Memory*;

			template<typename T, typename... Arguments>
			T* New(Arguments... parameters)
			{
				return new (allocate(sizeof(T))) T(parameters...);
			}

			template<typename T>
			void Delete(T* p)
			{
				reinterpret_cast<T*>(p)->~T();
				free(p, sizeof(T));
			}

			auto allocate(size_t size) noexcept -> void*;
			auto free(void* p, size_t size) noexcept -> void;

		private:
			static size_t* pBlockSizeLookup;
			static Allocator* pAllocators;

		private:
			static Allocator* lookUpAllocator(size_t size);
		};
	}
}