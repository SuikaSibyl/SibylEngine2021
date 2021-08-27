#pragma once
#include "SIByLpch.h"
#include "Sibyl/Renderer/CommandList.h"

namespace SIByL
{
	class DX12GraphicCommandList : public CommandList
	{
	public:
		DX12GraphicCommandList();
		~DX12GraphicCommandList()
		{
			m_GraphicCmdList = nullptr;
			m_CommandAllocator = nullptr;
		}

		virtual void Restart() override;
		virtual void Execute() override;

		inline ID3D12GraphicsCommandList* Get() { return m_GraphicCmdList.Get(); }

	private:
		ComPtr<ID3D12GraphicsCommandList>	m_GraphicCmdList;
		ComPtr<ID3D12CommandAllocator>		m_CommandAllocator;


		////////////////////////////////////////////////////////////////////////////////
		//                             Descriptor Binding                             //
		////////////////////////////////////////////////////////////////////////////////
	public:
		void SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE heapType, ComPtr<ID3D12DescriptorHeap> heap);
	private:
		// Keep track of the currently bound descriptor heaps. Only change descriptor 
		// heaps if they are different than the currently bound descriptor heaps.
		ComPtr<ID3D12DescriptorHeap> m_DescriptorHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES];
		// Binds the current descriptor heaps to the command list.
		void BindDescriptorHeaps();
	};
}