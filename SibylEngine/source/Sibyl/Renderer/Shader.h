#pragma once

#include "Sibyl/Renderer/VertexBuffer.h"
#include "Sibyl/Renderer/ShaderBinder.h"

namespace SIByL
{
	struct ShaderDesc
	{
		bool useAlphaBlending = false;
	};

	class Shader
	{
	public:
		static Shader* Create();
		static Shader* Create(std::string file, const ShaderDesc& desc = ShaderDesc());
		static Shader* Create(std::string vFile, std::string pFile, const ShaderDesc& desc = ShaderDesc());
		virtual void Use() = 0;
		virtual void CreateBinder(const VertexBufferLayout& vertexBufferLayout) = 0;
		virtual void SetVertexBufferLayout(const VertexBufferLayout& vertexBufferLayout) = 0;

		Ref<ShaderBinder> GetShaderBinder() { return m_ShaderBinder; }
	protected:
		Ref<ShaderBinder> m_ShaderBinder;
		ShaderDesc m_Descriptor;
	};
}