module;
#include <type_traits>
export module Core.SPointer;
import Core.Log;

namespace SIByL
{
	inline namespace Core
	{
		export
		template<typename T>
		class Scope
		{
		public:
			Scope();
			Scope(T* _ptr);
			Scope(Scope&& p);
			~Scope();

			T* get();
			T* operator->();

		private:
			// non-copyable
			Scope(const Scope& p) = delete;
			Scope& operator=(const Scope& p) = delete;

		private:
			T* ptr;
		};

		export
		template<typename T, typename... Args>
		Scope<T> CreateScope(Args&&... args)
		{
			auto res = new T(std::forward<Args>(args)...);
			if (res->initialize()) return Scope(res);
			else return Scope<T>();
		}

		template<typename T>
		Scope<T>::Scope() : ptr(nullptr)
		{}

		template<typename T>
		T* Scope<T>::get()
		{
			return ptr;
		}

		template<typename T>
		T* Scope<T>::operator->()
		{
			return ptr;
		}

		template<typename T>
		Scope<T>::Scope(T* _ptr) : ptr(_ptr)
		{}

		template<typename T>
		Scope<T>::Scope(Scope&& p) : ptr(p.ptr)
		{
			p.ptr = nullptr;
		}

		template<typename T>
		Scope<T>::~Scope()
		{
			if (ptr != nullptr)
			{
				if (ptr->destroy())
				{
					delete[] ptr;
					return;
				}
#ifdef _DEBUG
				SE_CORE_ERROR("SObject destroy failed!");
#endif // _DEBUG
			}
		}


	}
}