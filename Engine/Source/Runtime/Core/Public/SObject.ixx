module;
#include <type_traits>
export module Core.SObject;

namespace SIByL
{
	inline namespace Core
	{
		export class SObject
		{
		public:
			SObject() = default;
			virtual ~SObject() = default;
			virtual auto initialize() -> bool { return true; }
			virtual auto destroy() -> bool { return true; }
		};

		export class SFactory
		{
		public:
			template<typename T, typename... Args>
			inline static T* create(Args&&... args)
			{
				auto res = new T(std::forward<Args>(args)...);
				if (res->initialize()) return res;
				else return nullptr;
			}

			template<typename T>
			inline static bool destroy(T* object)
			{
				if (object->destroy())
				{
					delete object;
					return true;
				}
				return false;
			}
		};
	}
}