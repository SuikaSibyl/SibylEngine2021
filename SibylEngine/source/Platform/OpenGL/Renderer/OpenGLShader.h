#pragma once

#include "Sibyl/Renderer/Shader.h"

namespace SIByL
{
	class OpenGLShader :public Shader
	{
	public:
		OpenGLShader();
		virtual void Use() override;

	private:
		unsigned int m_ShaderProgram;

	};
}