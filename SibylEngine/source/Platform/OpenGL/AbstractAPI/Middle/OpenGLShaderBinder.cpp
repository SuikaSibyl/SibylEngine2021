#include "SIByLpch.h"
#include "OpenGLShaderBinder.h"

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Platform/OpenGL/Common/OpenGLContext.h"
#include "Platform/OpenGL/AbstractAPI/Middle/OpenGLTexture.h"

#include "Sibyl/Graphic/AbstractAPI/Library/ResourceLibrary.h"
#include "Sibyl/ECS/Core/SerializeUtility.h"
#include "Sibyl/ECS/Asset/AssetUtility.h"

namespace SIByL
{
	OpenGLShaderConstantsBuffer::OpenGLShaderConstantsBuffer(ShaderConstantsDesc* desc)
	{
		m_CpuBuffer = new byte[desc->Size];
		m_ConstantsMapper = &desc->Mapper;

		InitConstant();
	}

	OpenGLShaderConstantsBuffer::~OpenGLShaderConstantsBuffer()
	{
		delete[] m_CpuBuffer;
	}

	void OpenGLShaderConstantsBuffer::SetFloat(const std::string& name, const float& value)
	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value, item.Offset, ShaderDataTypeSize(item.Type));
		}
	}

	void OpenGLShaderConstantsBuffer::SetFloat3(const std::string& name, const glm::vec3& value)
	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value[0], item.Offset, ShaderDataTypeSize(item.Type));
		}
	}

	void OpenGLShaderConstantsBuffer::SetFloat4(const std::string& name, const glm::vec4& value)
	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value[0], item.Offset, ShaderDataTypeSize(item.Type));
		}
	}

	void OpenGLShaderConstantsBuffer::SetMatrix4x4(const std::string& name, const glm::mat4& value)
	{
		m_IsDirty = true;
		ShaderConstantItem item;
		if (m_ConstantsMapper->FetchConstant(name, item))
		{
			CopyMemoryToConstantsBuffer((void*)&value[0][0], item.Offset, ShaderDataTypeSize(item.Type));
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

		ShaderResourceItem item;
		if (m_ShaderResourcesDesc.Mapper.FetchResource(name, item))
		{
			m_ShaderResourcesDesc.Mapper.SetTextureID(name, texture->Identifer);
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
				Ref<Texture2D> refTex = Library<Texture2D>::Fetch(resource.second.TextureID);
				m_ShaderBinder->SetTexture2D(resource.first, refTex);
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
}
