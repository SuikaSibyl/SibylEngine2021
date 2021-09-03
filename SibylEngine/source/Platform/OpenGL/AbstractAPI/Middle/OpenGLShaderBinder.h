#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"

namespace SIByL
{
	class OpenGLShaderBinder :public ShaderBinder
	{
	public:
		virtual void SetFloat(const std::string& name, const float& value) override;
		virtual void SetFloat3(const std::string& name, const glm::vec3& value) override;
		virtual void SetFloat4(const std::string& name, const glm::vec4& value) override;
		virtual void SetMatrix4x4(const std::string& name, const glm::mat4& value) override;
		virtual void SetTexture2D(const std::string& name, Ref<Texture2D> texture) override;


		~OpenGLShaderBinder();

		OpenGLShaderBinder(const ShaderBinderDesc& desc);
		virtual void Bind() override {}

		void SetOpenGLShaderId(int id) { m_ShderID = id; }

		void TEMPUpdateAllConstants()
		{

		}
		void TEMPUpdateAllResources()
		{

		}

	private:
		int m_ShderID;
		ShaderBinderDesc m_Desc;
	};
}