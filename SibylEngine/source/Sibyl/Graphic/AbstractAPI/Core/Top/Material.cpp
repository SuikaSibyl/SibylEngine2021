#include "SIByLpch.h"
#include "Material.h"

#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h>

#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"
#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"

#include "yaml-cpp/yaml.h"
#include "Sibyl/ECS/Core/SerializeUtility.h"
#include "Sibyl/ECS/Asset/AssetUtility.h"

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
		m_Shader->UsePipelineState(pipelineStateDesc);

		m_ConstantsBuffer->UploadDataIfDirty(m_Shader->GetBinder().get());

		m_ResourcesBuffer->UploadDataIfDirty(m_Shader->GetBinder().get());

		m_CubeResourcesBuffer->UploadDataIfDirty(m_Shader->GetBinder().get());

	}

	////////////////////////////////////////////////////////////////////
	///					Parameter Setter / Getter					 ///
	void Material::SetFloat(const std::string& name, const float& value)
	{
		m_ConstantsBuffer->SetFloat(name, value);
		SetAssetDirty();
	}

	void Material::SetFloat3(const std::string& name, const glm::vec3& value)
	{
		m_ConstantsBuffer->SetFloat3(name, value);
		SetAssetDirty();
	}

	void Material::SetFloat4(const std::string& name, const glm::vec4& value)
	{
		m_ConstantsBuffer->SetFloat4(name, value);
		SetAssetDirty();
	}

	void Material::SetMatrix4x4(const std::string& name, const glm::mat4& value)
	{
		m_ConstantsBuffer->SetMatrix4x4(name, value);
		SetAssetDirty();
	}

	void Material::SetTexture2D(const std::string& name, Ref<Texture2D> texture)
	{
		m_ResourcesBuffer->SetTexture2D(name, texture);
		SetAssetDirty();
	}
	
	void Material::SetTexture2D(const std::string& name, RenderTarget* texture)
	{
		m_ResourcesBuffer->SetTexture2D(name, texture);
		SetAssetDirty();
	}

	void Material::SetTextureCubemap(const std::string& name, Ref<TextureCubemap> texture)
	{
		m_CubeResourcesBuffer->SetTextureCubemap(name, texture);
		SetAssetDirty();
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
		if (m_ResourcesBuffer != nullptr)
			return m_ResourcesBuffer->GetShaderResourceDesc();
		else
			return nullptr;
	}

	ShaderResourcesDesc* Material::GetCubeResourcesDesc()
	{
		if (m_CubeResourcesBuffer != nullptr)
			return m_CubeResourcesBuffer->GetShaderResourceDesc();
		else
			return nullptr;
	}

	////////////////////////////////////////////////////////////////////
	///							Custom Asset						 ///
	void Material::SaveAsset()
	{
		if (IsAssetDirty)
		{
			IsAssetDirty = false;
			Ref<Material> ref = Library<Material>::Fetch(SavePath);
			MaterialSerializer serializer(ref);
			serializer.Serialize("..\\Assets\\" + SavePath);
		}
	}

	void Material::UseShader(Ref<Shader> shader)
	{
		m_Shader = shader;
		m_ConstantsDesc = shader->GetBinder()->GetShaderConstantsDesc(1);
		m_ResourcesDesc = shader->GetBinder()->GetShaderResourcesDesc();
		m_CubeResourcesDesc = shader->GetBinder()->GetCubeShaderResourcesDesc();

		m_ConstantsBuffer = ShaderConstantsBuffer::Create
			(shader->GetBinder()->GetShaderConstantsDesc(1));
		m_ResourcesBuffer = ShaderResourcesBuffer::Create
			(shader->GetBinder()->GetShaderResourcesDesc(), 
			 shader->GetBinder()->GetRootSignature());
		m_CubeResourcesBuffer = ShaderResourcesBuffer::Create
			(shader->GetBinder()->GetCubeShaderResourcesDesc(),
			shader->GetBinder()->GetRootSignature());

		// Reset

		SetAssetDirty();
	}


	MaterialSerializer::MaterialSerializer(const Ref<Material>& material)
		:m_Material(material)
	{

	}

	void MaterialSerializer::Serialize(const std::string& filepath)
	{
		YAML::Emitter out;
		out << YAML::BeginMap;
		out << YAML::Key << "Material" << YAML::Value << "Untitled";

		std::string ShaderID = (m_Material->GetShaderUsed() == nullptr) ? "NONE" : m_Material->GetShaderUsed()->ShaderID;
		out << YAML::Key << "Shader" << YAML::Value << ShaderID;

		// Constants Buffer
		out << YAML::Key << "Constants Buffer" << YAML::Value << YAML::BeginSeq;
		ShaderConstantsDesc* desc = m_Material->GetConstantsDesc();
		int index = 0;
		if (desc != nullptr)
		{
			out << YAML::BeginMap;
			for (auto& item : *desc)
			{
				out << YAML::Key << "CONSTANT" << YAML::Value << index++;
				out << YAML::Key << "INFO";
				out << YAML::BeginMap;
				out << YAML::Key << "Type" << YAML::Value << (unsigned int)item.second.Type;
				out << YAML::Key << "Name" << YAML::Value << item.second.Name;
				switch (item.second.Type)
				{
				case ShaderDataType::RGBA:
				{
					glm::vec4 value;
					m_Material->GetFloat4(item.second.Name, value);
					out << YAML::Key << "Value" << YAML::Value << value;
					break;
				}
				default:
					break;
				}
				out << YAML::EndMap;
			}
			out << YAML::EndMap;
		}

		out << YAML::EndSeq;

		// Resources Buffer
		out << YAML::Key << "Textures" << YAML::Value << YAML::BeginSeq;
		ShaderResourcesDesc* resourcesDesc = m_Material->GetResourcesDesc();
		index = 0;
		if (resourcesDesc != nullptr)
		{
			for (auto& item : *resourcesDesc)
			{
				out << YAML::BeginMap;

				out << YAML::Key << "TEXTURE" << YAML::Value << index++;
				out << YAML::Key << "INFO";
				out << YAML::BeginMap;
				//out << YAML::Key << "Type" << YAML::Value << (unsigned int)item.second.Type;
				out << YAML::Key << "Name" << YAML::Value << item.second.Name;
				out << YAML::Key << "ID" << YAML::Value << item.second.TextureID;
				out << YAML::EndMap;

				out << YAML::EndMap;
			}
		}

		out << YAML::EndSeq;
		out << YAML::EndMap;

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
		std::ifstream stream("..\\Assets\\"+ filepath);
		std::stringstream strStream;
		strStream << stream.rdbuf();

		YAML::NodeAoS data = YAML::Load(strStream.str());
		if (!data["Material"])
			return false;

		// Set Shader
		std::string shaderName = data["Shader"].as<std::string>();
		if (shaderName != "NONE")
			m_Material->UseShader(Library<Shader>::Fetch(PathToIdentifier(shaderName)));

		// Set Constants
		auto constants = data["Constants Buffer"];
		if (constants)
		{
			for (auto constant : constants)
			{
				auto info = constant["INFO"];
				if (info)
				{
					int type = info["Type"].as<int>();
					std::string name = info["Name"].as<std::string>();
					switch (ShaderDataType(type))
					{
					case ShaderDataType::RGBA:
					case ShaderDataType::Float4:
					{
						glm::vec4 vector = info["Value"].as<glm::vec4>();
						m_Material->SetFloat4(name, vector);
						break;
					}
					default:
						break;
					}
				}
			}
		}

		// Set Textures
		auto textures = data["Textures"];
		if (textures)
		{
			for (auto texture : textures)
			{
				auto info = texture["INFO"];
				if (info)
				{
					//int type = info["Type"].as<int>();
					std::string name = info["Name"].as<std::string>();
					std::string id = info["ID"].as<std::string>();
					Ref<Texture2D> refTex = Library<Texture2D>::Fetch(id);
					m_Material->SetTexture2D(name, refTex);
				}
			}
		}

		m_Material->SetAssetUnDirty();

		return true;
	}
	bool MaterialSerializer::DeserializeRuntime(const std::string& filepath)
	{
		return false;
	}
}