#pragma once

#include "Sibyl/Renderer/VertexBuffer.h"
#include "Sibyl/Renderer/ShaderBinder.h"

namespace SIByL
{
	struct ShaderDesc
	{
		bool useAlphaBlending = false;
		VertexBufferLayout inputLayout;
	};

	class Shader
	{
	public:
		static Shader* Create();
		static Shader* Create(std::string file, const ShaderDesc& shaderDesc, const ShaderBinderDesc& binderDesc);
		static Shader* Create(std::string vFile, std::string pFile, const ShaderDesc& desc = ShaderDesc());
		virtual void Use() = 0;
		virtual void CreateBinder() = 0;
		virtual void SetVertexBufferLayout() = 0;

		Ref<ShaderBinder> GetShaderBinder() { return m_ShaderBinder; }
	protected:
		Ref<ShaderBinder> m_ShaderBinder;
		ShaderDesc m_Descriptor;
		ShaderBinderDesc m_BinderDescriptor;
	};
}