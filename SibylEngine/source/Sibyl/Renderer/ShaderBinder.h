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

	struct ShaderConstantItem
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

	class ConstantsMapper
	{

	private:
		std::unordered_map<std::string, ShaderConstantItem> m_Mapper;
	};

	class ShaderBinder
	{
	public:
		static Ref<ShaderBinder> Create(const ShaderBinderDesc& desc);
		virtual ~ShaderBinder() {}
		virtual void BindFloat3() = 0;
		virtual void Bind() = 0;

	protected:
		void InitMappers(const ShaderBinderDesc& desc);
		ConstantsMapper m_ConstantsMapper;
	};
}