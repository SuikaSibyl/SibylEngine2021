#pragma once

#include <glm/glm.hpp>

namespace SIByL
{
	class Material;
}

namespace SIByLEditor
{
	extern void DrawMaterial(const std::string& label, SIByL::Material& material);
}

namespace SIByL
{
	class Shader;
	class Texture2D;
	class ShaderConstantsBuffer;
	class ShaderResourcesBuffer;
	class ShaderConstantsDesc;
	class ShaderResourcesDesc;

	class Material
	{
	public:
		void SetPass();

		////////////////////////////////////////////////////////////////////
		///					Parameter Setter / Getter					 ///
		void SetFloat(const std::string& name, const float& value);
		void SetFloat3(const std::string& name, const glm::vec3& value);
		void SetFloat4(const std::string& name, const glm::vec4& value);
		void SetMatrix4x4(const std::string& name, const glm::mat4& value);
		void SetTexture2D(const std::string& name, Ref<Texture2D> texture);

		////////////////////////////////////////////////////////////////////
		///							Initializer							 ///
		Material() = default;
		Material(Ref<Shader> shader);
		void UseShader(Ref<Shader> shader);
		void OnDrawCall();

		////////////////////////////////////////////////////////////////////
		///							Fetcher								 ///
		ShaderConstantsDesc* GetConstantsDesc();
		ShaderResourcesDesc* GetResourcesDesc();

	private:
		Ref<Shader> m_Shader = nullptr;
		Ref<ShaderConstantsBuffer> m_ConstantsBuffer = nullptr;
		Ref<ShaderResourcesBuffer> m_ResourcesBuffer = nullptr;

		ShaderConstantsDesc* m_ConstantsDesc = nullptr;
		ShaderResourcesDesc* m_ResourcesDesc = nullptr;

		friend class DrawItem;
		friend void SIByLEditor::DrawMaterial(const std::string& label, SIByL::Material& material);
	};
}