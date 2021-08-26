#pragma once

#include "Sibyl/Renderer/ShaderBinder.h"
#include "Platform/DirectX12/Core/DynamicDescriptorHeap.h"
#include "Platform/DirectX12/Core/RootSignature.h"

namespace SIByL
{
	class DX12ShaderBinder :public ShaderBinder
	{
	public:
		DX12ShaderBinder();
		ID3D12RootSignature* GetRootSignature() { return m_RootSignature->GetRootSignature().Get(); }
		virtual void Bind() override;

		Ref<DynamicDescriptorHeap> GetSrvDynamicDescriptorHeap() { return m_SrvDynamicDescriptorHeap; }

	private:
		void BuildRootSignature();
		virtual void BindFloat3() override {}

	private:
		//ComPtr<ID3D12RootSignature> m_RootSignature;
		Ref<RootSignature> m_RootSignature;
		Ref<DynamicDescriptorHeap> m_SrvDynamicDescriptorHeap;
		Ref<DynamicDescriptorHeap> m_SamplerDynamicDescriptorHeap;
	};
}