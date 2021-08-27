#pragma once

#include "Sibyl/Renderer/Shader.h"

namespace SIByL
{
	class OpenGLShader :public Shader
	{
	public:
		OpenGLShader();
		OpenGLShader(std::string file, const ShaderDesc& desc);
		OpenGLShader(std::string vFile, std::string pFile, const ShaderDesc& desc);

		virtual void Use() override;
		virtual void CreateBinder() override;
		virtual void SetVertexBufferLayout() override;

	private:
		void CompileFromFile(std::string vertexPath, std::string fragmentPath);
		void CompileFromSingleFile(std::string glslPath);
		void CompileFromString(const char* vertex, const char* fragment);

	private:
		unsigned int m_ShaderProgram;
		VertexBufferLayout m_VertexBufferLayout;
	};
}