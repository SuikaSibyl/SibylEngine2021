module;
#include <string>
#include <filesystem>
export module Asset.ShaderLoader;
import Core.Buffer;
import Core.Cache;
import Core.MemoryManager;
import Asset.Asset;
import Asset.DedicatedLoader;
import Asset.Shader;
import RHI.IFactory;
import RHI.IShader;

namespace SIByL::Asset
{
	export struct ShaderLoader :public DedicatedLoader
	{
		ShaderLoader(Shader& shader) :shader(shader) {}
		ShaderLoader(Shader& shader, RHI::IResourceFactory* factory, RuntimeAssetManager* manager)
			:DedicatedLoader(factory, manager), shader(shader) {}

		virtual auto loadFromFile(std::filesystem::path path) noexcept -> void override;
		virtual auto loadFromCache(uint64_t const& path) noexcept -> void override;
		virtual auto saveAsCache(uint64_t const& path) noexcept -> void override;

		Shader& shader;
	};
	
	auto ShaderLoader::loadFromFile(std::filesystem::path path) noexcept -> void
	{
		resourceFactory->createShaderFromBinaryFile(path, { shader.stage,"main" });
	}

	auto ShaderLoader::loadFromCache(uint64_t const& path) noexcept -> void
	{
	}

	auto ShaderLoader::saveAsCache(uint64_t const& path) noexcept -> void
	{
	}
}