#include "SIByLpch.h"
#include "ComputeInstance.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLShaderBinder.h"

namespace SIByL
{
	void ComputeInstance::Dispatch(unsigned int x, unsigned int y, unsigned int z)
	{
		OnDrawCall();

		OpenGLShaderConstantsBuffer* SSBOBinder = dynamic_cast<OpenGLShaderConstantsBuffer*>(m_ConstantsBuffer.get());
		SSBOBinder->BindSSBO();
		m_Shader->Dispatch(x, y, z);
		SSBOBinder->UnbindSSBO();
	}

	////////////////////////////////////////////////////////////////////
	///					Parameter Setter / Getter					 ///
	void ComputeInstance::SetFloat(const std::string& name, const float& value)
	{
		m_ConstantsBuffer->SetFloat(name, value);
		SetAssetDirty();
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
		m_ResourcesBuffer->SetTexture2D(name, texture);
		SetAssetDirty();
	}

	void ComputeInstance::SetTexture2D(const std::string& name, RenderTarget* texture)
	{
		m_ResourcesBuffer->SetTexture2D(name, texture);
		SetAssetDirty();
	}

	void ComputeInstance::SetRenderTarget2D(const std::string& name, Ref<FrameBuffer> framebuffer, unsigned int attachmentIdx)
	{
		m_UnorderedAccessBuffer->SetRenderTarget2D(name, framebuffer, attachmentIdx);
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
		m_ConstantsDesc = shader->GetBinder()->GetShaderConstantsDesc(0);
		m_ResourcesDesc = shader->GetBinder()->GetShaderResourcesDesc();
		m_UnorderedAccessDesc = shader->GetBinder()->GetUnorderedAccessDesc();
		m_ConstantsBuffer = ShaderConstantsBuffer::Create
		(shader->GetBinder()->GetShaderConstantsDesc(0), true);
		m_ResourcesBuffer = ShaderResourcesBuffer::Create
		(shader->GetBinder()->GetShaderResourcesDesc(),
			shader->GetBinder()->GetRootSignature());
		m_UnorderedAccessBuffer = UnorderedAccessBuffer::Create
		(shader->GetBinder()->GetUnorderedAccessDesc(),
			shader->GetBinder()->GetRootSignature());
	}

	void ComputeInstance::OnDrawCall()
	{
		// Upload Per-Material parameters to GPU
		m_ConstantsBuffer->UploadDataIfDirty(m_Shader->GetBinder().get());
		m_ResourcesBuffer->UploadDataIfDirty(m_Shader->GetBinder().get());
		m_UnorderedAccessBuffer->UploadDataIfDirty(m_Shader->GetBinder().get());
	}

	////////////////////////////////////////////////////////////////////
	///							Serializer							 ///
	void ComputeInstance::SaveAsset()
	{

	}

}