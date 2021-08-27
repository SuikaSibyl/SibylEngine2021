#pragma once

#include "ShaderData.h"
#include "glm/glm.hpp"

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
		int ConstantBufferCount()const { return m_ConstantBufferLayouts.size(); }
		std::vector<ConstantBufferLayout> m_ConstantBufferLayouts;
		std::vector<ShaderResourceLayout> m_TextureBufferLayouts;
	};

	struct ShaderConstantItem
	{
		std::string Name;
		ShaderDataType Type;
		int CBIndex;
		uint32_t Offset;
	};

	class ConstantsMapper
	{
	public:
		void InsertConstant(const BufferElement& element, int CBIndex);
		bool FetchConstant(std::string name, ShaderConstantItem& buffer);

	private:
		std::unordered_map<std::string, ShaderConstantItem> m_Mapper;
	};

	struct ShaderResourceItem
	{
		std::string Name;
		ShaderResourceType Type;
	};

	class ResourcesMapper
	{
	public:
		//void InsertR
	private:
		std::unordered_map<std::string, ShaderResourceItem> m_Mapper;
	};

	class ShaderBinder
	{
	public:
		static Ref<ShaderBinder> Create(const ShaderBinderDesc& desc);
		virtual ~ShaderBinder() {}
		virtual void BindFloat3() = 0;
		virtual void Bind() = 0;

		virtual void SetFloat3(const std::string& name, const glm::vec3& value) = 0;
		virtual void TEMPUpdateAllConstants() = 0;
	protected:
		void InitMappers(const ShaderBinderDesc& desc);
		ConstantsMapper m_ConstantsMapper;
		ResourcesMapper m_ResourcesMapper;
	};
}