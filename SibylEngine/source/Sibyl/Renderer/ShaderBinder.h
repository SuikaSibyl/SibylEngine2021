#pragma once

#include "ShaderData.h"

namespace SIByL
{
	class ShaderBinderDesc
	{
	public:
		ShaderBinderDesc() = default;

		ShaderBinderDesc(
			const std::vector<ConstantBufferLayout>& cbLayouts,
			const std::vector<ShaderResourceLayout>& srLayouts ) 
			: m_ConstantBufferLayouts(cbLayouts)
			, m_TextureBufferLayouts(srLayouts) {}

		~ShaderBinderDesc() {};

		std::vector<ConstantBufferLayout> m_ConstantBufferLayouts;
		std::vector<ShaderResourceLayout> m_TextureBufferLayouts;
	};

	struct ShaderUniformItem
	{
		std::string Name;
		ShaderDataType Type;
	};

	struct ShaderConstants
	{

	};

	struct ShaderResourceTable
	{

	};

	class ShaderBinder
	{
	public:
		static ShaderBinder* Create(const ShaderBinderDesc& desc);
		virtual void BindFloat3() = 0;
		virtual void Bind() = 0;



	private:

	};
}