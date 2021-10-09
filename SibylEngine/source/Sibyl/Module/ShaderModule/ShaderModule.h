#pragma once

namespace SIByL
{
	class Shader;
	class Texture2D;
	class Material;
	class ShaderModule
	{
	public:
		static void Init();

		static Ref<Material> GetDefaultMaterial();

	private:
		static Ref<Shader>		DefaultShader;
		static Ref<Material>	DefaultMaterial;
		static Ref<Texture2D>	DefaultTexture;
	};
};