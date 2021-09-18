#pragma once

#include <glm/glm.hpp>
#include "Sibyl/ECS/Asset/CustomAsset.h"

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

	class Material :public CustomAsset
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

		void GetFloat(const std::string& name, float& value);
		void GetFloat3(const std::string& name, glm::vec3& value);
		void GetFloat4(const std::string& name, glm::vec4& value);
		void GetMatrix4x4(const std::string& name, glm::mat4& value);

		float* PtrFloat(const std::string& name);
		float* PtrFloat3(const std::string& name);
		float* PtrFloat4(const std::string& name);
		float* PtrMatrix4x4(const std::string& name);

		void SetDirty();

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
		Ref<Shader> GetShaderUsed() { return m_Shader; }

		////////////////////////////////////////////////////////////////////
		///						Custom Asset							 ///
		virtual void SaveAsset() override;

	private:
		Ref<Shader> m_Shader = nullptr;
		Ref<ShaderConstantsBuffer> m_ConstantsBuffer = nullptr;
		Ref<ShaderResourcesBuffer> m_ResourcesBuffer = nullptr;

		ShaderConstantsDesc* m_ConstantsDesc = nullptr;
		ShaderResourcesDesc* m_ResourcesDesc = nullptr;

		friend class DrawItem;
		friend class Camera;
		friend void SIByLEditor::DrawMaterial(const std::string& label, SIByL::Material& material);
	};

	class MaterialSerializer
	{
	public:
		MaterialSerializer(const Ref<Material>& material);

		void Serialize(const std::string& filepath);
		void SerializeRuntime(const std::string& filepath);

		bool Deserialize(const std::string& filepath);
		bool DeserializeRuntime(const std::string& filepath);

	private:
		Ref<Material> m_Material;
	};
}