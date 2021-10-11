#include "SIByLpch.h"
#include "ComputeInstance.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"

namespace SIByL
{
	////////////////////////////////////////////////////////////////////
	///					Parameter Setter / Getter					 ///
	void ComputeInstance::SetFloat(const std::string& name, const float& value)
	{

	}

	void ComputeInstance::SetFloat3(const std::string& name, const glm::vec3& value)
	{

	}

	void ComputeInstance::SetFloat4(const std::string& name, const glm::vec4& value)
	{

	}

	void ComputeInstance::SetMatrix4x4(const std::string& name, const glm::mat4& value)
	{

	}

	void ComputeInstance::SetTexture2D(const std::string& name, Ref<Texture2D> texture)
	{

	}

	void ComputeInstance::SetRenderTarget2D(const std::string& name, Ref<RenderTarget> rendertarget)
	{
		m_UnorderedAccessBuffer->SetRenderTarget2D(name, rendertarget);
		//m_ResourcesBuffer->SetTexture2D(name, texture);
		//SetAssetDirty();
	}


	////////////////////////////////////////////////////////////////////
	///							Initializer							 ///
	ComputeInstance::ComputeInstance(Ref<ComputeShader> shader)
	{
		UseShader(shader);
	}

	void ComputeInstance::UseShader(Ref<ComputeShader> shader)
	{
		m_Shader = shader;
		m_UnorderedAccessBuffer = UnorderedAccessBuffer::Create
		(shader->GetBinder()->GetUnorderedAccessDesc(),
			shader->GetBinder()->GetRootSignature());

	}

	void ComputeInstance::OnDrawCall()
	{

	}

}