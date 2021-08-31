#include "SIByLpch.h"
#include "OpenGLShaderBinder.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"
#include "Platform/OpenGL/Graphic/Texture/OpenGLTexture.h"

namespace SIByL
{
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
