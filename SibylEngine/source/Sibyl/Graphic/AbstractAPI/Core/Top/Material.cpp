#include "SIByLpch.h"
#include "Material.h"

#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h>
#include <Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h>

#include "Sibyl/Graphic/AbstractAPI/Core/Top/Graphic.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/Camera.h"

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
		//m_ConstantsBuffer->SetFloat(name, value);
	}

	////////////////////////////////////////////////////////////////////
	///							Initializer							 ///
	Material::Material(Ref<Shader> shader)
	{
		UseShader(shader);
	}

	void Material::UseShader(Ref<Shader> shader)
	{
		m_Shader = shader;
		m_ConstantsBuffer = ShaderConstantsBuffer::Create
			(shader->GetBinder()->GetShaderConstantsDesc(1));
	}

}