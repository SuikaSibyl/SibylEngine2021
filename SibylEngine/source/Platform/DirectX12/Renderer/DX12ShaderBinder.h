#pragma once

#include "Sibyl/Renderer/ShaderBinder.h"

namespace SIByL
{
	class DX12ShaderBinder :public ShaderBinder
	{
	public:
		DX12ShaderBinder();
		ID3D12RootSignature* GetRootSignature() { return m_RootSignature.Get(); }
		virtual void Bind() override;

		

	private:
		void BuildRootSignature();
		virtual void BindFloat3() override {}

	private:
		ComPtr<ID3D12RootSignature> m_RootSignature;
	};
}