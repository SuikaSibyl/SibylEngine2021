module;
#include <string>
export module ECS.TagComponent;
import ECS.Entity;

namespace SIByL::ECS
{
	export struct TagComponent
	{
		std::string Tag;
	};
}