#pragma once

#include "SIByLpch.h"
#include "OpenGLShader.h"

#include "Platform/OpenGL/Common/OpenGLContext.h"

namespace SIByL
{
	const char* vertexShaderSource = "#version 330 core\n"
		"layout (location = 0) in vec3 aPos;\n"
		"void main()\n"
		"{\n"
		"   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
		"}\0";

	const char* fragShaderSource = "#version 330 core\n"
		"out vec4 FragColor; \n"
		"void main()\n"
		"{\n"
		"FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f); \n"
		"}\0";

	OpenGLShader::OpenGLShader()
	{
		CompileFromString(vertexShaderSource, fragShaderSource);
	}

	OpenGLShader::OpenGLShader(std::string vFile, std::string pFile)
	{
		CompileFromFile(vFile, pFile);
	}

	void OpenGLShader::CompileFromFile(std::string vertexPath, std::string fragmentPath)
	{
		// 1. Get VERTEX/FRAGMENT From File Path
		std::string vertexCode;
		std::string fragmentCode;
		std::ifstream vShaderFile;
		std::ifstream fShaderFile;
		// Make sure exceptions could work properly
		vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		try
		{
			// Open the file
			vShaderFile.open(vertexPath);
			fShaderFile.open(fragmentPath);
			std::stringstream vShaderStream, fShaderStream;
			// Load the content from the File
			vShaderStream << vShaderFile.rdbuf();
			fShaderStream << fShaderFile.rdbuf();
			// Close File dealers
			vShaderFile.close();
			fShaderFile.close();
			// Change datastream to string
			vertexCode = vShaderStream.str();
			fragmentCode = fShaderStream.str();
		}
		catch (std::ifstream::failure e)
		{
			SIByL_CORE_ERROR("Shader File Failed to Load");
		}
		const char* vShaderCode = vertexCode.c_str();
		const char* fShaderCode = fragmentCode.c_str();
		CompileFromString(vShaderCode, fShaderCode);
	}

	void OpenGLShader::CompileFromString(const char* vertex, const char* fragment)
	{
		// ----------------------------
		// Create Vertex Shader
		// ----------------------------
		unsigned int vertexShader;
		vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, &vertex, NULL);
		glCompileShader(vertexShader);

		// Check Compiling Results
		int  success;
		char infoLog[512];
		glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
			SIByL_CORE_ERROR("Vertex Shader Compilation Failed : {0}", infoLog);
		}

		// ----------------------------
		// Create Fragment Shader
		// ----------------------------
		unsigned int fragmentShader;
		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, &fragment, NULL);
		glCompileShader(fragmentShader);

		// Check Compiling Results
		glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
			SIByL_CORE_ERROR("Fragment Shader Compilation Failed : {0}", infoLog);
		}

		// ----------------------------
		// Create Shader Program
		// ----------------------------
		m_ShaderProgram = glCreateProgram();
		glAttachShader(m_ShaderProgram, vertexShader);
		glAttachShader(m_ShaderProgram, fragmentShader);
		glLinkProgram(m_ShaderProgram);

		glGetProgramiv(m_ShaderProgram, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(m_ShaderProgram, 512, NULL, infoLog);
			SIByL_CORE_ERROR("Shader Link Failed : {0}", infoLog);
		}

		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);
	}

	void OpenGLShader::Use()
	{
		glUseProgram(m_ShaderProgram);
	}

	void OpenGLShader::CreateBinder(const VertexBufferLayout& vertexBufferLayout)
	{
		//m_ShaderBinder = 
	}

	void OpenGLShader::SetVertexBufferLayout(const VertexBufferLayout& vertexBufferLayout)
	{
		m_VertexBufferLayout = vertexBufferLayout;
	}
}