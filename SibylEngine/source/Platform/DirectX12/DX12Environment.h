#pragma once

#include "SIByLpch.h"

class DX12Environment
{
public:
	void Init();

public:
	inline ID3D12Device* GetDevice() { return m_D3dDevice.Get(); }

private:
	void CreateDevice();

private:
	ComPtr<IDXGIFactory4>	m_DxgiFactory;
	ComPtr<ID3D12Device>	m_D3dDevice;
};