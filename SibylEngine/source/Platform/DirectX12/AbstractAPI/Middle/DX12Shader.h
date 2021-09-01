#pragma once

#include "Sibyl/Graphic/AbstractAPI/Middle/Shader.h"

namespace SIByL
{
	class DX12Shader :public Shader
	{
	public:
		DX12Shader();
		DX12Shader(std::string file, const ShaderBinderDesc& binderDesc, const ShaderDesc& desc);
		DX12Shader(std::string vFile, std::string pFile, const ShaderDesc& desc);

		virtual void Use() override;
		virtual void CreateBinder() override;
		virtual void SetVertexBufferLayout() override;

	private:
		ComPtr<ID3DBlob> CompileFromFile(
			const std::wstring& fileName,
			const D3D_SHADER_MACRO* defines,
			const std::string& enteryPoint,
			const std::string& target);

		void CreatePSO();

	private:
		// Resources
		ComPtr<ID3DBlob> m_VsBytecode = nullptr;
		ComPtr<ID3DBlob> m_PsBytecode = nullptr;
		ComPtr<ID3D12PipelineState> m_PipelineStateObject;

		// Layout descriptors
		std::vector<D3D12_INPUT_ELEMENT_DESC> m_InputLayoutDesc;
		VertexBufferLayout m_VertexBufferLayout;
	};
}