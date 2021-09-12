#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"

namespace SIByL
{
	class OpenGLShaderConstantsBuffer :public ShaderConstantsBuffer
	{
	public:
		OpenGLShaderConstantsBuffer(ShaderConstantsDesc* desc);

		virtual void SetFloat(const std::string& name, const float& value) override;
		virtual void SetFloat3(const std::string& name, const glm::vec3& value) override;
		virtual void SetFloat4(const std::string& name, const glm::vec4& value) override;
		virtual void SetMatrix4x4(const std::string& name, const glm::mat4& value) override;

		virtual void GetFloat(const std::string& name, float& value) override;
		virtual void GetFloat3(const std::string& name, glm::vec3& value) override;
		virtual void GetFloat4(const std::string& name, glm::vec4& value) override;
		virtual void GetMatrix4x4(const std::string& name, glm::mat4& value) override;

		virtual float* PtrFloat(const std::string& name) override;
		virtual float* PtrFloat3(const std::string& name) override;
		virtual float* PtrFloat4(const std::string& name) override;
		virtual float* PtrMatrix4x4(const std::string& name) override; 

		virtual void UploadDataIfDirty() override;
		virtual void SetDirty() override;
	};

	class OpenGLShaderResourcesBuffer :public ShaderResourcesBuffer
	{
	public:
		OpenGLShaderResourcesBuffer(ShaderResourcesDesc* desc, RootSignature* rs);

		virtual void SetTexture2D(const std::string& name, Ref<Texture2D> texture) override;

		virtual void UploadDataIfDirty() override;

	private:
		ResourcesMapper* m_ResourcesMapper;
		bool m_IsDirty = true;
	};

	class OpenGLShaderBinder :public ShaderBinder
	{
	public:
		virtual void SetFloat(const std::string& name, const float& value);
		virtual void SetFloat3(const std::string& name, const glm::vec3& value);
		virtual void SetFloat4(const std::string& name, const glm::vec4& value);
		virtual void SetMatrix4x4(const std::string& name, const glm::mat4& value);
		virtual void SetTexture2D(const std::string& name, Ref<Texture2D> texture);

		virtual void BindConstantsBuffer(unsigned int slot, ShaderConstantsBuffer& buffer) {}

		~OpenGLShaderBinder();

		OpenGLShaderBinder(const ShaderBinderDesc& desc);
		virtual void Bind() override {}

		void SetOpenGLShaderId(int id) { m_ShderID = id; }

	private:
		int m_ShderID;
		ShaderBinderDesc m_Desc;
	};
}