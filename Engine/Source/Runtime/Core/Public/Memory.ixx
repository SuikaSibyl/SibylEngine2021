module;
#include <cstdint>
#include <utility>
#include <type_traits>
export module Core.MemoryManager;
import Core.Allocator;
import Core.Log;

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

			template<typename T, typename... Args>
			T* New(Args&&... args)
			{
				return ::new (allocate(sizeof(T))) T(std::forward<Args>(args)...);
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
			size_t* pBlockSizeLookup;
			Allocator* pAllocators;

		private:
			Allocator* lookUpAllocator(size_t size);
		};

		export auto MemAlloc(size_t size) noexcept -> void*
		{
			return Memory::instance()->allocate(size);
		}

		export auto MemFree(void* p, size_t size) noexcept -> void
		{
			Memory::instance()->free(p, size);
		}

		export
		template<typename T>
		class MemScope
		{
		public:
			MemScope();
			MemScope(T* _ptr);
			MemScope(MemScope&& p);
			~MemScope();

			T* get();
			void reset(T*);
			T* operator->();

			MemScope& operator=(MemScope&& p);

			template<typename T1, typename T2>
			friend MemScope<T1> MemCast(MemScope<T2>& input);

		private:
			// non-copyable
			MemScope(const MemScope& p) = delete;
			MemScope& operator=(const MemScope& p) = delete;
			// release
			void release();
			void regret();

		private:
			T* ptr;
		};

		export
		template<typename T, typename... Args>
		MemScope<T> CreateScope(Args&&... args)
		{
			auto res = new T(std::forward<Args>(args)...);
			if (res->initialize()) return Scope(res);
			else return Scope<T>();
		}

		template<typename T>
		MemScope<T>::MemScope() : ptr(nullptr)
		{}

		template<typename T>
		T* MemScope<T>::get()
		{
			return ptr;
		}

		template<typename T>
		void MemScope<T>::reset(T* _ptr)
		{
			release();
			ptr = _ptr;
		}

		template<typename T>
		T* MemScope<T>::operator->()
		{
			return ptr;
		}

		template<typename T>
		MemScope<T>::MemScope(T* _ptr) : ptr(_ptr)
		{}

		template<typename T>
		MemScope<T>::MemScope(MemScope&& p) : ptr(p.ptr)
		{
			p.ptr = nullptr;
		}

		template<typename T>
		MemScope<T>& MemScope<T>::operator=(MemScope<T>&& p)
		{
			reset(p.ptr);
			p.ptr = nullptr;
			return *this;
		}

		template<typename T>
		MemScope<T>::~MemScope()
		{
			release();
		}

		template<typename T>
		void MemScope<T>::release()
		{
			if (ptr != nullptr)
			{
				Memory::instance()->Delete(ptr);
				ptr = nullptr;
			}
		}

		template<typename T>
		void MemScope<T>::regret()
		{
			ptr = nullptr;
		}

		export
		template<typename T, typename... Args>
		MemScope<T> MemNew(Args&&... args)
		{
			T* ptr = Memory::instance()->New<T>(std::forward<Args>(args)...);
			return MemScope<T>(ptr);
		}


		export
		template<typename T1, typename T2>
		MemScope<T1> MemCast(MemScope<T2>& input)
		{
			MemScope<T1> res((T1*)input.get());
			input.regret();
			return res;
		}
	}
}