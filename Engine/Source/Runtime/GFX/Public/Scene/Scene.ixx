module;
#include <filesystem>
export module GFX.Scene;
import GFX.SceneTree;
import Core.File;

namespace SIByL::GFX
{
	export class Scene
	{
	public:
		auto serialize(std::filesystem::path path) noexcept -> void;

		SceneTree tree;
	};
}