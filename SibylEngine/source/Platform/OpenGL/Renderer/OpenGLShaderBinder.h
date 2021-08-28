#pragma once

#include "Sibyl/Renderer/ShaderBinder.h"

namespace SIByL
{
	class OpenGLShaderBinder :public ShaderBinder
	{
	public:
		~OpenGLShaderBinder();

		OpenGLShaderBinder(const ShaderBinderDesc& desc);
		virtual void Bind() override {}

		void SetOpenGLShaderId(int id) { m_ShderID = id; }

		virtual void SetFloat3(const std::string& name, const glm::vec3& value) override;
		virtual void SetTexture2D(const std::string& name, Ref<Texture2D> texture) override;

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