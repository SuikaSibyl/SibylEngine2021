#pragma once

#include "Sibyl/Renderer/ShaderBinder.h"
#include "Platform/DirectX12/Core/DynamicDescriptorHeap.h"
#include "Platform/DirectX12/Core/RootSignature.h"
#include "Platform/DirectX12/Core/DX12FrameResources.h"

namespace SIByL
{
	class DX12ShaderBinder :public ShaderBinder
	{
	public:
		~DX12ShaderBinder();

		DX12ShaderBinder(const ShaderBinderDesc& desc);
		ID3D12RootSignature* GetRootSignature() { return m_RootSignature->GetRootSignature().Get(); }
		virtual void Bind() override;

		virtual void SetFloat3(const std::string& name, const glm::vec3& value) override;

		Ref<DynamicDescriptorHeap> GetSrvDynamicDescriptorHeap() { return m_SrvDynamicDescriptorHeap; }

		void TEMPUpdateAllConstants()
		{
			UpdateConstantsBuffer(0);
			BindConstantsBuffer(0);
		}

	private:
		void BuildRootSignature();
		virtual void BindFloat3() override {}

	private:
		Ref<RootSignature> m_RootSignature;
		Ref<DynamicDescriptorHeap> m_SrvDynamicDescriptorHeap;
		Ref<DynamicDescriptorHeap> m_SamplerDynamicDescriptorHeap;
		ShaderBinderDesc m_Desc;

		Ref<DX12FrameResourceBuffer>* m_ConstantsTableBuffer;
		void CopyMemoryToConstantsBuffer(void* data, int index, uint32_t offset, uint32_t length);
		void UpdateConstantsBuffer(int index);
		void BindConstantsBuffer(int index);
	};
}