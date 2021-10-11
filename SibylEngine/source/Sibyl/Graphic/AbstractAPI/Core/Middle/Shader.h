#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/VertexBuffer.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"

namespace SIByL
{
	struct ShaderDesc
	{
		bool useAlphaBlending = false;
		VertexBufferLayout inputLayout;
		int NumRenderTarget = 1;
	};

	class Shader
	{
	public:
		static Ref<Shader> Create();
		static Ref<Shader> Create(std::string file, const ShaderDesc& shaderDesc, const ShaderBinderDesc& binderDesc);
		static Ref<Shader> Create(std::string vFile, std::string pFile, const ShaderDesc& desc = ShaderDesc());

		virtual ~Shader() {}
		virtual void Use() = 0;
		virtual void CreateBinder() = 0;
		virtual void SetVertexBufferLayout() = 0;

		Ref<ShaderBinder> GetBinder() { return m_ShaderBinder; }
		std::string ShaderID;

	protected:
		Ref<ShaderBinder> m_ShaderBinder;
		ShaderDesc m_Descriptor;
		ShaderBinderDesc m_BinderDescriptor;
	};

	class ComputeShader
	{
	public:
		static Ref<ComputeShader> Create(std::string file, const ShaderBinderDesc& binderDesc);

		virtual void Dispatch(unsigned int x, unsigned int y, unsigned int z) = 0;
		virtual void CreateBinder() = 0;

		Ref<ShaderBinder> GetBinder() { return m_ShaderBinder; }
		std::string ShaderID;

	protected:
		Ref<ShaderBinder> m_ShaderBinder;
		ShaderBinderDesc m_BinderDescriptor;
	};
}