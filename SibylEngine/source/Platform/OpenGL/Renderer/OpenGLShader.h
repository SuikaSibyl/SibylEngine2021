#pragma once

#include "Sibyl/Renderer/Shader.h"

namespace SIByL
{
	class OpenGLShader :public Shader
	{
	public:
		OpenGLShader();
		OpenGLShader(std::string vFile, std::string pFile);
		virtual void Use() override;

	private:
		void CompileFromFile(std::string vertexPath, std::string fragmentPath);
		void CompileFromString(const char* vertex, const char* fragment);

	private:
		unsigned int m_ShaderProgram;

	};
}