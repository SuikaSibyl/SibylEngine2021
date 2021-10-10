#pragma once

#include "SIByLpch.h"
#include "DX12Shader.h"

#pragma comment(lib, "D3D12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "d3dcompiler.lib")

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12ShaderBinder.h"

namespace SIByL
{
	static DXGI_FORMAT ShaderDataTypeToDXGIFormat(ShaderDataType type)
	{
		switch (type)
		{
		case SIByL::ShaderDataType::None:	return DXGI_FORMAT_R8_TYPELESS;
		case SIByL::ShaderDataType::Float:	return DXGI_FORMAT_R32_FLOAT;
		case SIByL::ShaderDataType::Float2:	return DXGI_FORMAT_R32G32_FLOAT;
		case SIByL::ShaderDataType::Float3:	return DXGI_FORMAT_R32G32B32_FLOAT;
		case SIByL::ShaderDataType::Float4:	return DXGI_FORMAT_R32G32B32A32_FLOAT;
		case SIByL::ShaderDataType::Mat3:	return DXGI_FORMAT_R8_TYPELESS;
		case SIByL::ShaderDataType::Mat4:	return DXGI_FORMAT_R8_TYPELESS;
		case SIByL::ShaderDataType::Int:	return DXGI_FORMAT_R16_SINT;
		case SIByL::ShaderDataType::Int2:	return DXGI_FORMAT_R8G8_SINT;
		case SIByL::ShaderDataType::Int3:	return DXGI_FORMAT_R32G32B32_SINT;
		case SIByL::ShaderDataType::Int4:	return DXGI_FORMAT_R8G8B8A8_SINT;
		case SIByL::ShaderDataType::Bool:	return DXGI_FORMAT_R8_TYPELESS;
		case SIByL::ShaderDataType::RGB:	return DXGI_FORMAT_R32G32B32_FLOAT;
		case SIByL::ShaderDataType::RGBA:	return DXGI_FORMAT_R32G32B32A32_FLOAT;
		default:return DXGI_FORMAT_R8_TYPELESS;
		}
	}

	DX12Shader::DX12Shader()
	{
	}

	DX12Shader::DX12Shader(std::string file, const ShaderBinderDesc& binderDesc, const ShaderDesc& desc)
	{
		PROFILE_SCOPE_FUNCTION();

		m_Descriptor = desc;
		m_BinderDescriptor = binderDesc;
		m_VertexBufferLayout = desc.inputLayout;

		m_VsBytecode = CompileFromFile(AnsiToWString(file), nullptr, "VS", "vs_5_1");
		m_PsBytecode = CompileFromFile(AnsiToWString(file), nullptr, "PS", "ps_5_1");

		CreateBinder();
	}
	
	DX12Shader::DX12Shader(std::string vFile, std::string pFile, const ShaderDesc& desc)
	{
		PROFILE_SCOPE_FUNCTION();

		m_Descriptor = desc;
		m_VsBytecode = CompileFromFile(AnsiToWString(vFile), nullptr, "VS", "vs_5_1");
		m_PsBytecode = CompileFromFile(AnsiToWString(pFile), nullptr, "PS", "ps_5_1");
	}

	void DX12Shader::Use()
	{
		PROFILE_SCOPE_FUNCTION();

		// Set PSO
		ID3D12GraphicsCommandList* cmdList = DX12Context::GetInFlightDXGraphicCommandList();
		cmdList->SetPipelineState(m_PipelineStateObject.Get());
		// Bind Stuff
		m_ShaderBinder->Bind();
	}

	void DX12Shader::CreateBinder()
	{
		PROFILE_SCOPE_FUNCTION();

		m_ShaderBinder = ShaderBinder::Create(m_BinderDescriptor);
		SetVertexBufferLayout();
		CreatePSO();
	}

	void DX12Shader::SetVertexBufferLayout()
	{
		PROFILE_SCOPE_FUNCTION();

		for (const auto& element : m_VertexBufferLayout)
		{
			m_InputLayoutDesc.push_back(
				{ element.Name.c_str(), 0,
				ShaderDataTypeToDXGIFormat(element.Type) , 0, element.Offset,
				D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 });
		}
	}

	ComPtr<ID3DBlob> DX12Shader::CompileFromFile(
		const std::wstring& fileName,
		const D3D_SHADER_MACRO* defines,
		const std::string& enteryPoint,
		const std::string& target)
	{
		PROFILE_SCOPE_FUNCTION();

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
			OutputDebugStringA((char*)errors->GetBufferPointer());
		}
		DXCall(hr);

		return byteCode;
	}

	void DX12Shader::CreatePSO()
	{
		PROFILE_SCOPE_FUNCTION();

		DX12ShaderBinder* dxShaderBinder = dynamic_cast<DX12ShaderBinder*>(m_ShaderBinder.get());
		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
		ZeroMemory(&psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
		psoDesc.InputLayout = { m_InputLayoutDesc.data(), (UINT)m_InputLayoutDesc.size() };
		psoDesc.pRootSignature = dxShaderBinder->GetDXRootSignature();
		psoDesc.VS = { reinterpret_cast<BYTE*>(m_VsBytecode->GetBufferPointer()), m_VsBytecode->GetBufferSize() };
		psoDesc.PS = { reinterpret_cast<BYTE*>(m_PsBytecode->GetBufferPointer()), m_PsBytecode->GetBufferSize() };
		psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
		psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
		psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
		psoDesc.SampleMask = UINT_MAX;	//0xffffffff, No Sampling Mask
		psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
		psoDesc.NumRenderTargets = m_Descriptor.NumRenderTarget;
		psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;	// Normalized Unsigned Int
		psoDesc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
		psoDesc.SampleDesc.Count = 1;	// No 4XMSAA
		psoDesc.SampleDesc.Quality = 0;	////No 4XMSAA

		DXCall(DX12Context::GetDevice()->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_PipelineStateObject)));
	}
}