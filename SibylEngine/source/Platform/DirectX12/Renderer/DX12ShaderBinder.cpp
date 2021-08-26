#include "SIByLpch.h"
#include "DX12ShaderBinder.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/Core/UploadBuffer.h"

namespace SIByL
{
	DX12ShaderBinder::DX12ShaderBinder()
	{
		BuildRootSignature();
		m_SrvDynamicDescriptorHeap = std::make_shared<DynamicDescriptorHeap>(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
		m_SamplerDynamicDescriptorHeap = std::make_shared<DynamicDescriptorHeap>(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

		m_SrvDynamicDescriptorHeap->ParseRootSignature(*m_RootSignature);
	}

	void DX12ShaderBinder::Bind()
	{
		// Bind Root Signature
		ID3D12GraphicsCommandList* cmdList = DX12Context::GetDXGraphicCommandList();
		cmdList->SetGraphicsRootSignature(GetRootSignature());

		//// Bind Descriptor Table
		//int objCbvIndex = 0;
		//auto handle = CD3DX12_GPU_DESCRIPTOR_HANDLE(cbvHeap->GetGPUDescriptorHandleForHeapStart());
		//handle.Offset(objCbvIndex, cbv_srv_uavDescriptorSize);
		//cmdList->SetGraphicsRootDescriptorTable(0, //根参数的起始索引
		//	handle);
	}

	void DX12ShaderBinder::BuildRootSignature()
	{
		CD3DX12_DESCRIPTOR_RANGE srvTableCube;
		srvTableCube.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV,	//描述符类型
			1,	//表中的描述符数量（纹理数量）
			0);	//描述符所绑定的寄存器槽号

		// Perfomance TIP: Order from most frequent to least frequent.
		// ----------------------------------------------------------------------
		
		// RootSignature could be descriptor table \ Root Descriptor \ Root Constant
		CD3DX12_ROOT_PARAMETER slotRootParameter[3];
		// Create a descriptor table of one sigle CBV
		slotRootParameter[0].InitAsConstantBufferView(0);
		slotRootParameter[1].InitAsDescriptorTable(1,//Range数量
			&srvTableCube,	//Range指针
			D3D12_SHADER_VISIBILITY_PIXEL);	//该资源只能在像素着色器可读

		auto staticSamplers = DX12Context::GetStaticSamplers();	//获得静态采样器集合
		//slotRootParameter[0].InitAsDescriptorTable(1, &cbvTable);
		// Root Signature is consisted of a set of root parameters
		CD3DX12_ROOT_SIGNATURE_DESC rootSig(2, // Number of root parameters
			slotRootParameter, // Pointer to Root Parameter
			staticSamplers.size(),
			staticSamplers.data(),
			D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
		// Create a root signature using a single register
		// The slot is pointing to a descriptor area where is only on econst buffer
		ComPtr<ID3DBlob> serializedRootSig = nullptr;
		ComPtr<ID3DBlob> errorBlob = nullptr;
		HRESULT hr = D3D12SerializeRootSignature(&rootSig, D3D_ROOT_SIGNATURE_VERSION_1, &serializedRootSig, &errorBlob);

		if (errorBlob != nullptr)
		{
			OutputDebugStringA((char*)errorBlob->GetBufferPointer());
		}
		DXCall(hr);

		m_RootSignature = std::make_shared<RootSignature>(rootSig);
	}

}