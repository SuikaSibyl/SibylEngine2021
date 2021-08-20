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

	private:
		ComPtr<ID3DBlob> CompileFromFile(
			const std::wstring& fileName,
			const D3D_SHADER_MACRO* defines,
			const std::string& enteryPoint,
			const std::string& target);

	private:
		ComPtr<ID3DBlob> m_VsBytecode = nullptr;
		ComPtr<ID3DBlob> m_PsBytecode = nullptr;
	};
}