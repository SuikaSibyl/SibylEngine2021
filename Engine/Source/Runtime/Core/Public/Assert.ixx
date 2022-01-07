module;
#include <utility>
export module Core.Assert;
import Core.Log;

export template <class... Args> auto SE_CORE_ASSERT(bool x, Args&&... args) noexcept -> void 
{
	if (!(x))
	{
		SE_CORE_ERROR(std::forward<Args>(args)...);
		__debugbreak();
	}
}

export template <class... Args> auto SE_ASSERT(bool x, Args&&... args) noexcept -> void
{
	if (!(x))
	{
		SE_ERROR(std::forward<Args>(args)...);
		__debugbreak();
	}
}