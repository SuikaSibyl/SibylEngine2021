#include "SIByLpch.h"
#include "OpenGLShaderBinder.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/FrameBufferTexture.h"
#include "Sibyl/Graphic/AbstractAPI/Library/FrameBufferLibrary.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLTexture.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLFrameBufferTexture.h"

#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/ECS/Core/SerializeUtility.h"
#include "Sibyl/ECS/Asset/AssetUtility.h"

namespace SIByL
{
	OpenGLShaderConstantsBuffer::OpenGLShaderConstantsBuffer(ShaderConstantsDesc* desc, bool isSSBO)
	{
		m_CpuBuffer = new byte[desc->Size];
		m_ConstantsMapper = &desc->Mapper;
		m_IsSSBO = isSSBO;
		InitConstant();
		SSBOSize = sizeof(byte) * desc->Size;
		if (m_IsSSBO)
		{
			glGenBuffers(1, &SSBO);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
			glBufferData(GL_SHADER_STORAGE_BUFFER, SSBOSize, m_CpuBuffer, GL_DYNAMIC_DRAW);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, SSBO);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}
	}
	void OpenGLShaderConstantsBuffer::BindSSBO()
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, SSBO);
		glBufferData(GL_SHADER_STORAGE_BUFFER, SSBOSize, m_CpuBuffer, GL_DYNAMIC_DRAW);
	}

	void OpenGLShaderConstantsBuffer::UnbindSSBO()
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	OpenGLShaderConstantsBuffer::~OpenGLShaderConstantsBuffer()
	{
		delete[] m_CpuBuffer;
	}

	void OpenGLShaderConstantsBuffer::SetInt(const std::string& name, const int& value)
	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value, item.Offset, ShaderDataTypeSize(item.Type));
			if (m_IsSSBO)
			{
				BindSSBO();
				glBufferSubData(GL_SHADER_STORAGE_BUFFER, item.Offset, ShaderDataTypeSize(item.Type), (void*)&value);
				UnbindSSBO();
			}
		}
	}
	void OpenGLShaderConstantsBuffer::SetFloat(const std::string& name, const float& value)
	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value, item.Offset, ShaderDataTypeSize(item.Type));
			if (m_IsSSBO)
			{
				BindSSBO();
				glBufferSubData(GL_SHADER_STORAGE_BUFFER, item.Offset, ShaderDataTypeSize(item.Type), (void*)&value);
				UnbindSSBO();
			}
		}
	}
	void OpenGLShaderConstantsBuffer::SetFloat2(const std::string& name, const glm::vec2& value)

	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value[0], item.Offset, ShaderDataTypeSize(item.Type));
			if (m_IsSSBO)
			{
				BindSSBO();
				glBufferSubData(GL_SHADER_STORAGE_BUFFER, item.Offset, ShaderDataTypeSize(item.Type), (void*)&value[0]);
				UnbindSSBO();
			}
		}
	}

	void OpenGLShaderConstantsBuffer::SetFloat3(const std::string& name, const glm::vec3& value)
	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value[0], item.Offset, ShaderDataTypeSize(item.Type));
			if (m_IsSSBO)
			{
				BindSSBO();
				glBufferSubData(GL_SHADER_STORAGE_BUFFER, item.Offset, ShaderDataTypeSize(item.Type), (void*)&value[0]);
				UnbindSSBO();
			}
		}
	}

	void OpenGLShaderConstantsBuffer::SetFloat4(const std::string& name, const glm::vec4& value)
	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value[0], item.Offset, ShaderDataTypeSize(item.Type));
			if (m_IsSSBO)
			{
				BindSSBO();
				glBufferSubData(GL_SHADER_STORAGE_BUFFER, item.Offset, ShaderDataTypeSize(item.Type), (void*)&value[0]);
				UnbindSSBO();
			}
		}
	}

	void OpenGLShaderConstantsBuffer::SetMatrix4x4(const std::string& name, const glm::mat4& value)
	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value[0][0], item.Offset, ShaderDataTypeSize(item.Type));
			if (m_IsSSBO)
			{
				BindSSBO();
				glBufferSubData(GL_SHADER_STORAGE_BUFFER, item.Offset, ShaderDataTypeSize(item.Type), (void*)&value[0][0]);
				UnbindSSBO();
			}
		}
	}

	void OpenGLShaderConstantsBuffer::GetInt(const std::string& name, int& value)
	{
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryFromConstantsBuffer((void*)&value, item.Offset, ShaderDataTypeSize(item.Type));
		}
	}

	void OpenGLShaderConstantsBuffer::GetFloat(const std::string& name, float& value)
	{
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryFromConstantsBuffer((void*)&value, item.Offset, ShaderDataTypeSize(item.Type));
		}
	}

	void OpenGLShaderConstantsBuffer::GetFloat3(const std::string& name, glm::vec3& value)
	{
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryFromConstantsBuffer((void*)&value[0], item.Offset, ShaderDataTypeSize(item.Type));
		}
	}

	void OpenGLShaderConstantsBuffer::GetFloat4(const std::string& name, glm::vec4& value)
	{
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryFromConstantsBuffer((void*)&value[0], item.Offset, ShaderDataTypeSize(item.Type));
		}
	}

	void OpenGLShaderConstantsBuffer::GetMatrix4x4(const std::string& name, glm::mat4& value)
	{
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryFromConstantsBuffer((void*)&value[0][0], item.Offset, ShaderDataTypeSize(item.Type));
		}
	}

	float* OpenGLShaderConstantsBuffer::PtrFloat(const std::string& name)
	{
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			return (float*)GetPtrFromConstantsBuffer(item.Offset, ShaderDataTypeSize(item.Type));
		}
		return nullptr;
	}

	float* OpenGLShaderConstantsBuffer::PtrFloat3(const std::string& name)
	{
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			return (float*)GetPtrFromConstantsBuffer(item.Offset, ShaderDataTypeSize(item.Type));
		}
		return nullptr;
	}

	float* OpenGLShaderConstantsBuffer::PtrFloat4(const std::string& name)
	{
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			return (float*)GetPtrFromConstantsBuffer(item.Offset, ShaderDataTypeSize(item.Type));
		}
		return nullptr;
	}

	float* OpenGLShaderConstantsBuffer::PtrMatrix4x4(const std::string& name)
	{
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			return (float*)GetPtrFromConstantsBuffer(item.Offset, ShaderDataTypeSize(item.Type));
		}
		return nullptr;
	}

	void OpenGLShaderConstantsBuffer::UploadDataIfDirty(ShaderBinder* shaderBinder)
	{
		OpenGLShaderBinder* m_ShaderBinder = dynamic_cast<OpenGLShaderBinder*>(shaderBinder);
		for (auto iter : *m_ConstantsMapper)
		{
			switch (iter.second.Type)
			{
			case ShaderDataType::Int:
			{
				int value;
				GetInt(iter.second.Name, value);
				m_ShaderBinder->SetInt(iter.second.Name, value);
				break;
			}
			case ShaderDataType::Float:
			{
				float value;
				GetFloat(iter.second.Name, value);
				m_ShaderBinder->SetFloat(iter.second.Name, value);
				break;
			}
			case ShaderDataType::RGB:
			case ShaderDataType::Float3:
			{
				glm::vec3 value;
				GetFloat3(iter.second.Name, value);
				m_ShaderBinder->SetFloat3(iter.second.Name, value);
				break;
			}
			case ShaderDataType::RGBA:
			case ShaderDataType::Float4:
			{
				glm::vec4 value;
				GetFloat4(iter.second.Name, value);
				m_ShaderBinder->SetFloat4(iter.second.Name, value);
				break;
			}
			case ShaderDataType::Mat4:
			{
				glm::mat4 value;
				GetMatrix4x4(iter.second.Name, value);
				m_ShaderBinder->SetMatrix4x4(iter.second.Name, value);
				break;
			}
			}
		}
		/*m_ShaderBinder*/
	}

	void OpenGLShaderConstantsBuffer::SetDirty()
	{
		m_IsDirty = true;
	}

	void OpenGLShaderConstantsBuffer::InitConstant()
	{
		for each (auto & constant in *m_ConstantsMapper)
		{
			switch (constant.second.Type)
			{
			case ShaderDataType::RGBA:
				SetFloat4(constant.first, { 0, 0, 0, 1 });
				break;
			case ShaderDataType::Float4:
				SetFloat4(constant.first, { 0, 0, 0, 0 });
				break;

			default:
				break;
			}
		}
	}

	void OpenGLShaderConstantsBuffer::CopyMemoryToConstantsBuffer
	(void* data, uint32_t offset, uint32_t length)
	{
		void* target = (void*)((char*)m_CpuBuffer + offset);
		memcpy((target)
			, data
			, length);
	}

	void OpenGLShaderConstantsBuffer::CopyMemoryFromConstantsBuffer
	(void* data, uint32_t offset, uint32_t length)
	{
		void* target = (void*)((char*)m_CpuBuffer + offset);
		memcpy(data
			, (target)
			, length);
	}

	void* OpenGLShaderConstantsBuffer::GetPtrFromConstantsBuffer
	(uint32_t offset, uint32_t length)
	{
		return (void*)((char*)m_CpuBuffer + offset);
	}

	OpenGLShaderResourcesBuffer::OpenGLShaderResourcesBuffer(ShaderResourcesDesc* desc, RootSignature* rs)
	{
		m_ShaderResourcesDesc = *desc;
	}
	
	ShaderResourcesDesc* OpenGLShaderResourcesBuffer::GetShaderResourceDesc()
	{
		return &m_ShaderResourcesDesc;
	}

	void OpenGLShaderResourcesBuffer::SetTexture2D(const std::string& name, Ref<Texture2D> texture)
	{
		m_IsDirty = true;

		if (texture == nullptr)return;
		ShaderResourceItem item;
		if (m_ShaderResourcesDesc.Mapper.FetchResource(name, item))
		{
			m_ShaderResourcesDesc.Mapper.SetTextureID(name, texture->Identifer);
		}
	}

	void OpenGLShaderResourcesBuffer::SetTexture2D(const std::string& name, RenderTarget* texture)
	{
		m_IsDirty = true;

		ShaderResourceItem item;
		if (m_ShaderResourcesDesc.Mapper.FetchResource(name, item))
		{
			m_ShaderResourcesDesc.Mapper.SetTextureID(name, texture->GetIdentifier(), ShaderResourceType::RenderTarget);
		}
	}
	
	void OpenGLShaderResourcesBuffer::SetTextureCubemap(const std::string& name, Ref<TextureCubemap> texture)
	{
		m_IsDirty = true;

		ShaderResourceItem item;
		if (m_ShaderResourcesDesc.Mapper.FetchResource(name, item))
		{
			m_ShaderResourcesDesc.Mapper.SetTextureID(name, texture->Identifer, ShaderResourceType::Cubemap);
		}
	}

	void OpenGLShaderResourcesBuffer::UploadDataIfDirty(ShaderBinder* shaderBinder)
	{
		if (true)
		{
			m_IsDirty = false;

			OpenGLShaderBinder* m_ShaderBinder = dynamic_cast<OpenGLShaderBinder*>(shaderBinder);

			for each (auto & resource in m_ShaderResourcesDesc.Mapper)
			{
				switch (resource.second.Type)
				{
				case ShaderResourceType::RenderTarget:
				{
					RenderTarget* rendertarget = FrameBufferLibrary::GetRenderTarget(resource.second.TextureID);
					OpenGLRenderTarget* oglprocrt = dynamic_cast<OpenGLRenderTarget*>(rendertarget);
					oglprocrt->SetShaderResource(resource.second.Offset);
				}
					break;
				case ShaderResourceType::Texture2D:
				{
					Ref<Texture2D> refTex = Library<Texture2D>::Fetch(resource.second.TextureID);
					m_ShaderBinder->SetTexture2D(resource.first, refTex);
				}
				break;
				case ShaderResourceType::Cubemap:
				{
					Ref<TextureCubemap> refTex = Library<TextureCubemap>::Fetch(resource.second.TextureID);
					m_ShaderBinder->SetTextureCubemap(resource.first, refTex);

				}
					break;
				default:
					break;
				}
			}
		}
	}

	OpenGLUnorderedAccessBuffer::OpenGLUnorderedAccessBuffer(ShaderResourcesDesc* desc, RootSignature* rs)
	{
		m_ShaderResourcesDesc = *desc;
	}

	void OpenGLUnorderedAccessBuffer::SetTexture2D(const std::string& name, Ref<Texture2D> texture)
	{
		m_IsDirty = true;

		ShaderResourceItem item;
		if (m_ShaderResourcesDesc.Mapper.FetchResource(name, item))
		{
			m_ShaderResourcesDesc.Mapper.SetTextureID(name, texture->Identifer);
		}
	}

	void OpenGLUnorderedAccessBuffer::SetRenderTarget2D(const std::string& name, Ref<FrameBuffer> framebuffer, unsigned int attachmentIdx, unsigned int mip)
	{
		m_IsDirty = true;

		ShaderResourceItem item;
		if (m_ShaderResourcesDesc.Mapper.FetchResource(name, item))
		{
			m_ShaderResourcesDesc.Mapper.SetTextureID(name, framebuffer->GetIdentifier() + std::to_string(attachmentIdx), ShaderResourceType::RenderTarget, mip);
		}
	}

	ShaderResourcesDesc* OpenGLUnorderedAccessBuffer::GetShaderResourceDesc()
	{
		return nullptr;
	}

	void OpenGLUnorderedAccessBuffer::UploadDataIfDirty(ShaderBinder* shaderBinder)
	{
		if (true)
		{
			m_IsDirty = false;

			OpenGLShaderBinder* m_ShaderBinder = dynamic_cast<OpenGLShaderBinder*>(shaderBinder);

			for each (auto & resource in m_ShaderResourcesDesc.Mapper)
			{
				RenderTarget* rendertarget = FrameBufferLibrary::GetRenderTarget(resource.second.TextureID);
				m_ShaderBinder->SetRenderTarget2D(resource.first, rendertarget, resource.second.SelectedMip);
			}
		}
	}


	OpenGLShaderBinder::~OpenGLShaderBinder()
	{

	}

	OpenGLShaderBinder::OpenGLShaderBinder(const ShaderBinderDesc& desc)
	{
		PROFILE_SCOPE_FUNCTION();

		InitMappers(desc);

	}
	void OpenGLShaderBinder::SetOpenGLShaderId(int id)
	{
		m_ShderID = id;

		glUseProgram(m_ShderID);
		for (auto& resourceSlot : m_ResourcesMapper)
		{
			unsigned int location = glGetUniformLocation(m_ShderID, resourceSlot.second.Name.c_str());
			glUniform1i(location, resourceSlot.second.Offset);
		}
		for (auto& resourceSlot : m_CubeResourcesMapper)
		{
			unsigned int location = glGetUniformLocation(m_ShderID, resourceSlot.second.Name.c_str());
			glUniform1i(location, resourceSlot.second.Offset);
		}
		glUseProgram(0);
	}

	void OpenGLShaderBinder::SetInt(const std::string& name, const int& value)
	{
		glUniform1iv(glGetUniformLocation(m_ShderID, name.c_str()), 1, &value);
	}

	void OpenGLShaderBinder::SetFloat(const std::string& name, const float& value)
	{
		PROFILE_SCOPE_FUNCTION();

		glUniform1fv(glGetUniformLocation(m_ShderID, name.c_str()), 1, &value);
	}

	void OpenGLShaderBinder::SetFloat3(const std::string& name, const glm::vec3& value)
	{
		PROFILE_SCOPE_FUNCTION();

		glUniform3fv(glGetUniformLocation(m_ShderID, name.c_str()), 1, &value[0]);
	}

	void OpenGLShaderBinder::SetFloat4(const std::string& name, const glm::vec4& value)
	{
		PROFILE_SCOPE_FUNCTION();

		glUniform4fv(glGetUniformLocation(m_ShderID, name.c_str()), 1, &value[0]);
	}

	void OpenGLShaderBinder::SetMatrix4x4(const std::string& name, const glm::mat4& value)
	{
		PROFILE_SCOPE_FUNCTION();

		glUniformMatrix4fv(glGetUniformLocation(m_ShderID, name.c_str()), 1, GL_FALSE, &value[0][0]);
	}

	void OpenGLShaderBinder::SetTexture2D(const std::string& name, Ref<Texture2D> texture)
	{
		PROFILE_SCOPE_FUNCTION();

		ShaderResourceItem item;
		if (m_ResourcesMapper.FetchResource(name, item))
		{
			OpenGLTexture2D* oglTexture = dynamic_cast<OpenGLTexture2D*>(texture.get());
			oglTexture->Bind(item.Offset);
		}
	}

	void OpenGLShaderBinder::SetTextureCubemap(const std::string& name, Ref<TextureCubemap> texture)
	{
		ShaderResourceItem item;
		if (m_CubeResourcesMapper.FetchResource(name, item))
		{
			OpenGLTextureCubemap* oglTexture = dynamic_cast<OpenGLTextureCubemap*>(texture.get());
			if (oglTexture == nullptr) return;
			oglTexture->Bind(item.Offset);
		}
	}

	void OpenGLShaderBinder::SetRenderTarget2D(const std::string& name, RenderTarget* rendertarget, unsigned int miplevel)
	{
		ShaderResourceItem item;
		if (m_UnorderedAccessMapper.FetchResource(name, item))
		{
			OpenGLRenderTarget* rt = dynamic_cast<OpenGLRenderTarget*>(rendertarget);
			rt->SetComputeRenderTarget(item.Offset, miplevel);
		}
	}
}
