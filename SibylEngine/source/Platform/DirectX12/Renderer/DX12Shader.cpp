#pragma once

#include "SIByLpch.h"
#include "DX12Shader.h"

#pragma comment(lib, "D3D12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

namespace SIByL
{
	DX12Shader::DX12Shader()
	{

	}

	DX12Shader::DX12Shader(std::string vFile, std::string pFile)
	{
		m_VsBytecode = CompileFromFile(AnsiToWString(vFile), nullptr, "VS", "vs_5_1");
		m_PsBytecode = CompileFromFile(AnsiToWString(pFile), nullptr, "PS", "ps_5_1");
	}

	void DX12Shader::Use()
	{

	}

	ComPtr<ID3DBlob> DX12Shader::CompileFromFile(
		const std::wstring& fileName,
		const D3D_SHADER_MACRO* defines,
		const std::string& enteryPoint,
		const std::string& target)
	{
		// If in debug mode, compile using DEBUG Mode
		UINT compileFlags = 0;
#if defined(DEBUG) || defined(_DEBUG)
		compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;
#endif // defined(DEBUG) || defined(_DEBUG)
		
		HRESULT hr = S_OK;

		ComPtr<ID3DBlob> byteCode = nullptr;
		ComPtr<ID3DBlob> errors;
		hr = D3DCompileFromFile(fileName.c_str(), //hlsl源文件名
			defines,	//高级选项，指定为空指针
			D3D_COMPILE_STANDARD_FILE_INCLUDE,	//高级选项，可以指定为空指针
			enteryPoint.c_str(),	//着色器的入口点函数名
			target.c_str(),		//指定所用着色器类型和版本的字符串
			compileFlags,	//指示对着色器断代码应当如何编译的标志
			0,	//高级选项
			&byteCode,	//编译好的字节码
			&errors);	//错误信息

		if (errors != nullptr)
		{
			SIByL_CORE_ERROR("DX12 Shader Compile From File Failed!");
		}
		DXCall(hr);

		return byteCode;
	}
}