#include "SIByLpch.h"
#include "DX12Environment.h"
#include "DX12Utility.h"

void DX12Environment::Init()
{
	CreateDevice();
}

void DX12Environment::CreateDevice()
{
	DXCall(CreateDXGIFactory1(IID_PPV_ARGS(&m_DxgiFactory)));
	DXCall(D3D12CreateDevice(nullptr,
		D3D_FEATURE_LEVEL_12_0,
		IID_PPV_ARGS(&m_D3dDevice)));
}