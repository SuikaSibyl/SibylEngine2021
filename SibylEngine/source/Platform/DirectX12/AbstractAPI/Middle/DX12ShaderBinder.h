#pragma once

#include "Sibyl/Graphic/AbstractAPI/Core/Middle/ShaderBinder.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12DynamicDescriptorHeap.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12RootSignature.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12FrameResources.h"

namespace SIByL
{
	class DX12ShaderBinder :public ShaderBinder
	{
	public:
		virtual void SetFloat(const std::string& name, const float& value) override;
		virtual void SetFloat3(const std::string& name, const glm::vec3& value) override;
		virtual void SetFloat4(const std::string& name, const glm::vec4& value) override;
		virtual void SetMatrix4x4(const std::string& name, const glm::mat4& value) override;
		virtual void SetTexture2D(const std::string& name, Ref<Texture2D> texture) override;

		~DX12ShaderBinder();

		DX12ShaderBinder(const ShaderBinderDesc& desc);
		ID3D12RootSignature* GetRootSignature() { return m_RootSignature->GetRootSignature().Get(); }
		virtual void Bind() override;
		Ref<DX12DynamicDescriptorHeap> GetSrvDynamicDescriptorHeap() { return m_SrvDynamicDescriptorHeap; }

		void TEMPUpdateAllConstants()
		{
			UpdateConstantsBuffer(0);
			BindConstantsBuffer(0);
		}
		void TEMPUpdateAllResources()
		{
			GetSrvDynamicDescriptorHeap()->CommitStagedDescriptorsForDraw();
		}

	private:
		void BuildRootSignature();

	private:
		Ref<RootSignature> m_RootSignature;
		Ref<DX12DynamicDescriptorHeap> m_SrvDynamicDescriptorHeap;
		Ref<DX12DynamicDescriptorHeap> m_SamplerDynamicDescriptorHeap;
		ShaderBinderDesc m_Desc;

		Ref<DX12FrameResourceBuffer>* m_ConstantsTableBuffer;
		void CopyMemoryToConstantsBuffer(void* data, int index, uint32_t offset, uint32_t length);
		void UpdateConstantsBuffer(int index);
		void BindConstantsBuffer(int index);
	};
}