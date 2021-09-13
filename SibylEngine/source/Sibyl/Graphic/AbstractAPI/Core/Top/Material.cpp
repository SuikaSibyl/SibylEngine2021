#include "SIByLpch.h"
#include "Material.h"

#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h>

#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"

#include "yaml-cpp/yaml.h"

namespace SIByL
{
	void Material::SetPass()
	{
		// Current Material
		Graphic::CurrentMaterial = this;
		// Use Shader
		m_Shader->Use();
		// Bind Per-Material parameters to Shader
		m_Shader->GetBinder()->BindConstantsBuffer(1, *m_ConstantsBuffer);
		m_Shader->GetBinder()->BindConstantsBuffer(2, Graphic::CurrentCamera->GetConstantsBuffer());
	}

	void Material::OnDrawCall()
	{
		// Upload Per-Material parameters to GPU
		m_ConstantsBuffer->UploadDataIfDirty();

		m_ResourcesBuffer->UploadDataIfDirty();
	}

	////////////////////////////////////////////////////////////////////
	///					Parameter Setter / Getter					 ///
	void Material::SetFloat(const std::string& name, const float& value)
	{
		m_ConstantsBuffer->SetFloat(name, value);
	}

	void Material::SetFloat3(const std::string& name, const glm::vec3& value)
	{
		m_ConstantsBuffer->SetFloat3(name, value);
	}

	void Material::SetFloat4(const std::string& name, const glm::vec4& value)
	{
		m_ConstantsBuffer->SetFloat4(name, value);
	}

	void Material::SetMatrix4x4(const std::string& name, const glm::mat4& value)
	{
		m_ConstantsBuffer->SetMatrix4x4(name, value);
	}

	void Material::SetTexture2D(const std::string& name, Ref<Texture2D> texture)
	{
		m_ResourcesBuffer->SetTexture2D(name, texture);
	}

	void Material::GetFloat(const std::string& name, float& value)
	{
		m_ConstantsBuffer->SetFloat(name, value);
	}

	void Material::GetFloat3(const std::string& name, glm::vec3& value)
	{
		m_ConstantsBuffer->GetFloat3(name, value);
	}

	void Material::GetFloat4(const std::string& name, glm::vec4& value)
	{
		m_ConstantsBuffer->GetFloat4(name, value);
	}

	void Material::GetMatrix4x4(const std::string& name, glm::mat4& value)
	{
		m_ConstantsBuffer->GetMatrix4x4(name, value);
	}

	float* Material::PtrFloat(const std::string& name)
	{
		return m_ConstantsBuffer->PtrFloat(name);
	}

	float* Material::PtrFloat3(const std::string& name)
	{
		return m_ConstantsBuffer->PtrFloat3(name);
	}

	float* Material::PtrFloat4(const std::string& name)
	{
		return m_ConstantsBuffer->PtrFloat4(name);
	}

	float* Material::PtrMatrix4x4(const std::string& name)
	{
		return m_ConstantsBuffer->PtrMatrix4x4(name);
	}

	void Material::SetDirty()
	{
		m_ConstantsBuffer->SetDirty();
	}

	////////////////////////////////////////////////////////////////////
	///							Initializer							 ///
	Material::Material(Ref<Shader> shader)
	{
		UseShader(shader);
	}

	////////////////////////////////////////////////////////////////////
	///							Fetcher								 ///
	ShaderConstantsDesc* Material::GetConstantsDesc()
	{
		return m_ConstantsDesc;
	}

	ShaderResourcesDesc* Material::GetResourcesDesc()
	{
		return m_ResourcesDesc;
	}

	void Material::UseShader(Ref<Shader> shader)
	{
		m_Shader = shader;
		m_ConstantsDesc = shader->GetBinder()->GetShaderConstantsDesc(1);
		m_ResourcesDesc = shader->GetBinder()->GetShaderResourcesDesc();
		m_ConstantsBuffer = ShaderConstantsBuffer::Create
			(shader->GetBinder()->GetShaderConstantsDesc(1));
		m_ResourcesBuffer = ShaderResourcesBuffer::Create
			(shader->GetBinder()->GetShaderResourcesDesc(), 
			 shader->GetBinder()->GetRootSignature());
	}


	MaterialSerializer::MaterialSerializer(const Ref<Material>& material)
		:m_Material(material)
	{

	}

	void MaterialSerializer::Serialize(const std::string& filepath)
	{
		YAML::Emitter out;
		out << YAML::BeginMap;
		out << YAML::Key << "Material" << YAML::Value << "Unamed";
		out << YAML::Key << "Constants Buffer" << YAML::Value << YAML::BeginSeq;
		
		ShaderConstantsDesc* desc = m_Material->GetConstantsDesc();
		for (auto& item : *desc)
		{

		}
		//m_Scene->m_Registry.each([&](auto entityID)
		//	{
		//		Entity entity = { entityID, m_Scene.get() };
		//		if (!entity)
		//			return;

		//		SerializeEntity(out, entity);
		//	});

		out << YAML::EndSeq;
		out << YAML::EndMap;

		std::ofstream fout(filepath);
		fout << out.c_str();

	}
	void MaterialSerializer::SerializeRuntime(const std::string& filepath)
	{

	}

	bool MaterialSerializer::Deserialize(const std::string& filepath)
	{
		return true;
	}
	bool MaterialSerializer::DeserializeRuntime(const std::string& filepath)
	{
		return false;
	}
}