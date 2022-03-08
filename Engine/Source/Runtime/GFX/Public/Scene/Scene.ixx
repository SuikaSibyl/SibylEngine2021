module;
#include <filesystem>
export module GFX.Scene;
import GFX.SceneTree;
import Core.File;
import RHI.ILogicalDevice;

namespace SIByL::GFX
{
	export class Scene
	{
	public:
		auto serialize(std::filesystem::path path) noexcept -> void;
		auto deserialize(std::filesystem::path path, RHI::ILogicalDevice* device) noexcept -> void;

		SceneTree tree;
	};
}