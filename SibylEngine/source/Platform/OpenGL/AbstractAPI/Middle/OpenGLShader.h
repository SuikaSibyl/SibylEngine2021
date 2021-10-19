#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Shader.h"

namespace SIByL
{
	class OpenGLShader :public Shader
	{
	public:
		OpenGLShader();
		OpenGLShader(std::string file, const ShaderDesc& shaderDesc, const ShaderBinderDesc& binderDesc);
		OpenGLShader(std::string vFile, std::string pFile, const ShaderDesc& desc);

		virtual void Use() override;
		virtual void CreateBinder() override;
		virtual void SetVertexBufferLayout() override;
		virtual void UsePipelineState(const PipelineStateDesc& desc) override;

	private:
		void CompileFromFile(std::string vertexPath, std::string fragmentPath);
		void CompileFromSingleFile(std::string glslPath);
		void CompileFromString(const char* vertex, const char* fragment);

	private:
		//ShaderDesc m_Descriptor;
		unsigned int m_ShaderProgram;
		VertexBufferLayout m_VertexBufferLayout;
		ShaderBinderDesc m_BinderDescriptor;
	};

	class OpenGLComputeShader :public ComputeShader
	{
	public:
		OpenGLComputeShader(std::string file, const ShaderBinderDesc& binderDesc);

		virtual void Dispatch(unsigned int x, unsigned int y, unsigned int z) override;
		virtual void CreateBinder() override;

	private:
		void CompileFromFile(std::string shaderPath);
		void CompileFromString(const char* shaderString);

	private:
		unsigned int m_ShaderProgram;
	};
}