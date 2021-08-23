#pragma once
#include "SIByLpch.h"
#include "Sibyl/Renderer/CommandList.h"

namespace SIByL
{
	class DX12GraphicCommandList : public CommandList
	{
	public:
		DX12GraphicCommandList();

		virtual void Restart() override;
		virtual void Execute() override;

		inline ID3D12GraphicsCommandList* Get() { return m_GraphicCmdList.Get(); }

	private:
		ComPtr<ID3D12GraphicsCommandList>	m_GraphicCmdList;
		ComPtr<ID3D12CommandAllocator>		m_CommandAllocator;

	};
}