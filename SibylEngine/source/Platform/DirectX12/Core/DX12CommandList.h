#pragma once
#include "SIByLpch.h"

namespace SIByL
{
	class DX12GraphicCommandList
	{
	public:
		DX12GraphicCommandList();

		void Restart();
		void Execute();

		inline ID3D12GraphicsCommandList* Get() { return m_GraphicCmdList.Get(); }

	private:
		ComPtr<ID3D12GraphicsCommandList>	m_GraphicCmdList;
		ComPtr<ID3D12CommandAllocator>		m_CommandAllocator;

	};
}