#pragma once

#include "Sibyl/Renderer/VertexBuffer.h"
#include "Sibyl/Renderer/ShaderBinder.h"

namespace SIByL
{
	class Shader
	{
	public:
		static Shader* Create();
		static Shader* Create(std::string vFile, std::string pFile);
		virtual void Use() = 0;
		virtual void CreateBinder(const VertexBufferLayout& vertexBufferLayout) = 0;
		virtual void SetVertexBufferLayout(const VertexBufferLayout& vertexBufferLayout) = 0;

	protected:
		std::unique_ptr<ShaderBinder> m_ShaderBinder;
	};
}