#pragma once

#include "ShaderData.h"
#include "glm/glm.hpp"
#include "Sibyl/Graphic/AbstractAPI/Core/Middle/Texture.h"

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
		size_t ConstantBufferCount()const { return m_ConstantBufferLayouts.size(); }
		size_t TextureBufferCount()const { return m_TextureBufferLayouts.size(); }
		std::vector<ConstantBufferLayout> m_ConstantBufferLayouts;
		std::vector<ShaderResourceLayout> m_TextureBufferLayouts;
	};

	/// <summary>
	/// Shader Constant Mapper
	/// </summary>
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
		//m_Mapper.begin()
		using iterator = std::unordered_map<std::string, ShaderConstantItem>::iterator;
		iterator begin() { return m_Mapper.begin(); }
		iterator end() { return m_Mapper.end(); }

	private:
		std::unordered_map<std::string, ShaderConstantItem> m_Mapper;
	};

	/// <summary>
	/// Shader Resource Mapper
	/// </summary>
	struct ShaderResourceItem
	{
		std::string Name;
		ShaderResourceType Type;
		int SRTIndex;
		int Offset;
	};
	class ResourcesMapper
	{
	public:
		void InsertResource(const ShaderResourceItem& element);
		bool FetchResource(std::string name, ShaderResourceItem& buffer);
		//m_Mapper.begin()
		using iterator = std::unordered_map<std::string, ShaderResourceItem>::iterator;
		iterator begin() { return m_Mapper.begin(); }
		iterator end() { return m_Mapper.end(); }

	private:
		std::unordered_map<std::string, ShaderResourceItem> m_Mapper;
	};

	//////////////////////////////////////////////
	///			Shader Constants Buffer			//
	//////////////////////////////////////////////
	struct ShaderConstantsDesc
	{
		uint32_t Size = -1;
		ConstantsMapper Mapper;

		ConstantsMapper::iterator begin() { return Mapper.begin(); }
		ConstantsMapper::iterator end() { return Mapper.end(); }
	};
	class ShaderBinder;
	class ShaderConstantsBuffer
	{
	public:
		static Ref<ShaderConstantsBuffer> Create(ShaderConstantsDesc* desc);
		virtual ~ShaderConstantsBuffer() = default;

		virtual void SetFloat(const std::string& name, const float& value) = 0;
		virtual void SetFloat3(const std::string& name, const glm::vec3& value) = 0;
		virtual void SetFloat4(const std::string& name, const glm::vec4& value) = 0;
		virtual void SetMatrix4x4(const std::string& name, const glm::mat4& value) = 0;

		virtual void GetFloat(const std::string& name, float& value) = 0;
		virtual void GetFloat3(const std::string& name, glm::vec3& value) = 0;
		virtual void GetFloat4(const std::string& name, glm::vec4& value) = 0;
		virtual void GetMatrix4x4(const std::string& name, glm::mat4& value) = 0;

		virtual float* PtrFloat(const std::string& name) = 0;
		virtual float* PtrFloat3(const std::string& name) = 0;
		virtual float* PtrFloat4(const std::string& name) = 0;
		virtual float* PtrMatrix4x4(const std::string& name) = 0;


		virtual void UploadDataIfDirty(ShaderBinder* m_ShaderBinder) = 0;
		virtual void SetDirty() = 0;

	protected:
		virtual void InitConstant() = 0;
	};
	//////////////////////////////////////////////

	//////////////////////////////////////////////
	///			Shader Resource Buffer			//
	//////////////////////////////////////////////
	struct ShaderResourcesDesc
	{
		uint32_t Size = -1;
		ResourcesMapper Mapper;

		ResourcesMapper::iterator begin() { return Mapper.begin(); }
		ResourcesMapper::iterator end() { return Mapper.end(); }
	};
	class RootSignature;
	class ShaderResourcesBuffer
	{
	public:
		static Ref<ShaderResourcesBuffer> Create(ShaderResourcesDesc* desc, RootSignature* rs);
		virtual ~ShaderResourcesBuffer() = default;

		virtual void SetTexture2D(const std::string& name, Ref<Texture2D> texture) = 0;

		virtual void UploadDataIfDirty() = 0;
	};

	//////////////////////////////////////////////

	//////////////////////////////////////////////
	///				Shader Binder				//
	//////////////////////////////////////////////
	class ShaderDescriptorTableBuffer
	{

	};

	class ShaderBinder
	{
	public:
		static Ref<ShaderBinder> Create(const ShaderBinderDesc& desc);
		virtual ~ShaderBinder() { delete[] m_ShaderConstantDescs; }


		virtual void BindConstantsBuffer(unsigned int slot, ShaderConstantsBuffer& buffer) = 0;
		virtual ShaderConstantsDesc* GetShaderConstantsDesc(unsigned int slot) { return &m_ShaderConstantDescs[slot]; }
		virtual ShaderResourcesDesc* GetShaderResourcesDesc() { return &m_ShaderResourceDescs; }
		virtual RootSignature* GetRootSignature() { return nullptr; }

		virtual void Bind() = 0;

	protected:
		void InitMappers(const ShaderBinderDesc& desc);
		ConstantsMapper m_ConstantsMapper;
		ResourcesMapper m_ResourcesMapper;
		ShaderConstantsDesc* m_ShaderConstantDescs;
		ShaderResourcesDesc m_ShaderResourceDescs;
	};
	//////////////////////////////////////////////
}