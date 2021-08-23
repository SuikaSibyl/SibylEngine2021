#pragma once

#include "Sibyl/Renderer/Shader.h"

namespace SIByL
{
	class DX12Shader :public Shader
	{
	public:
		DX12Shader();
		DX12Shader(std::string vFile, std::string pFile);

		virtual void Use() override;
		virtual void CreateBinder(const VertexBufferLayout& vertexBufferLayout) override;
		virtual void SetVertexBufferLayout(const VertexBufferLayout& vertexBufferLayout) override;

	private:
		ComPtr<ID3DBlob> CompileFromFile(
			const std::wstring& fileName,
			const D3D_SHADER_MACRO* defines,
			const std::string& enteryPoint,
			const std::string& target);

		void CreatePSO();

	private:
		ComPtr<ID3DBlob> m_VsBytecode = nullptr;
		ComPtr<ID3DBlob> m_PsBytecode = nullptr;
		ComPtr<ID3D12PipelineState> m_PipelineStateObject;
		std::vector<D3D12_INPUT_ELEMENT_DESC> m_InputLayoutDesc;
		VertexBufferLayout m_VertexBufferLayout;
	};
}