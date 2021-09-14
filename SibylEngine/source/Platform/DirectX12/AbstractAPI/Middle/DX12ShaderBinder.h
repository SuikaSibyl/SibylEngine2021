#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12DynamicDescriptorHeap.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12RootSignature.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12FrameResources.h"

namespace SIByL
{
	//////////////////////////////////////////////
	///			Shader Constants Buffer			//
	//////////////////////////////////////////////
	class DX12ShaderConstantsBuffer :public ShaderConstantsBuffer
	{
	public:
		DX12ShaderConstantsBuffer(ShaderConstantsDesc* desc);
		
		virtual void SetFloat(const std::string& name, const float& value) override;
		virtual void SetFloat3(const std::string& name, const glm::vec3& value) override;
		virtual void SetFloat4(const std::string& name, const glm::vec4& value) override;
		virtual void SetMatrix4x4(const std::string& name, const glm::mat4& value) override;

		virtual void GetFloat(const std::string& name, float& value) override;
		virtual void GetFloat3(const std::string& name, glm::vec3& value) override;
		virtual void GetFloat4(const std::string& name, glm::vec4& value) override;
		virtual void GetMatrix4x4(const std::string& name, glm::mat4& value) override;

		virtual float* PtrFloat(const std::string& name) override;
		virtual float* PtrFloat3(const std::string& name) override;
		virtual float* PtrFloat4(const std::string& name) override;
		virtual float* PtrMatrix4x4(const std::string& name) override;

		virtual void UploadDataIfDirty(ShaderBinder* shaderBinder) override;
		virtual void SetDirty() override;


	private:
		friend class DX12ShaderBinder;
		Ref<DX12FrameResourceBuffer> m_ConstantsTableBuffer = nullptr;
		ConstantsMapper* m_ConstantsMapper;
		bool m_IsDirty = true;
	};	

	//////////////////////////////////////////////

	//////////////////////////////////////////////
	///			Shader Resource Buffer			//
	//////////////////////////////////////////////
	class DX12ShaderResourcesBuffer :public ShaderResourcesBuffer
	{
	public:
		DX12ShaderResourcesBuffer(ShaderResourcesDesc* desc, RootSignature* rootsignature);
		
		virtual void SetTexture2D(const std::string& name, Ref<Texture2D> texture) override;

		virtual void UploadDataIfDirty() override;

	private:
		friend class DX12ShaderBinder;

		ResourcesMapper* m_ResourcesMapper;
		Ref<DX12DynamicDescriptorHeap> m_SrvDynamicDescriptorHeap;
		Ref<DX12DynamicDescriptorHeap> m_SamplerDynamicDescriptorHeap;

		bool m_IsDirty = true;
	};

	//////////////////////////////////////////////

	//////////////////////////////////////////////
	///				Shader Binder				//
	//////////////////////////////////////////////
	class DX12ShaderBinder :public ShaderBinder
	{
	public:
		////////////////////////////////////////////////////////////////////
		///					Constructors/Desctructors					 ///
		DX12ShaderBinder(const ShaderBinderDesc& desc);
		~DX12ShaderBinder();

		////////////////////////////////////////////////////////////////////
		///								Bind							 ///
		virtual void BindConstantsBuffer(unsigned int slot, ShaderConstantsBuffer& buffer) override;

		virtual void SetTexture2D(const std::string& name, Ref<Texture2D> texture);

		ID3D12RootSignature* GetDXRootSignature() { return m_RootSignature->GetRootSignature().Get(); }
		RootSignature* GetRootSignature() { return m_RootSignature.get(); }
		virtual void Bind() override;
		Ref<DX12DynamicDescriptorHeap> GetSrvDynamicDescriptorHeap() { return m_SrvDynamicDescriptorHeap; }

	private:
		void BuildRootSignature();

	private:
		Ref<DX12RootSignature> m_RootSignature;
		Ref<DX12DynamicDescriptorHeap> m_SrvDynamicDescriptorHeap;
		Ref<DX12DynamicDescriptorHeap> m_SamplerDynamicDescriptorHeap;
		ShaderBinderDesc m_Desc;
	};
}