#include "SIByLpch.h"
#include "DX12CommandList.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

#include "DX12FrameResources.h"

namespace SIByL
{
	DX12GraphicCommandList::DX12GraphicCommandList()
	{
		DXCall(DX12Context::GetDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_CommandAllocator)));
		DXCall(DX12Context::GetDevice()->CreateCommandList(0,
			D3D12_COMMAND_LIST_TYPE_DIRECT,
			m_CommandAllocator.Get(),
			nullptr,
			IID_PPV_ARGS(&m_GraphicCmdList)));
		m_GraphicCmdList->Close();
	}

	void DX12GraphicCommandList::Restart()
	{
		ComPtr<ID3D12CommandAllocator> commandAllocator = DX12FrameResourcesManager::GetCurrentAllocator();
		DXCall(commandAllocator->Reset());
		DXCall(m_GraphicCmdList->Reset(commandAllocator.Get(), nullptr));

		// INIT
		BindDescriptorHeaps();
	}

	void DX12GraphicCommandList::Execute()
	{
		DXCall(m_GraphicCmdList->Close());
		ID3D12CommandList* cmdLists[] = { m_GraphicCmdList.Get() };
		DX12Context::GetCommandQueue()->ExecuteCommandLists(_countof(cmdLists), cmdLists);
	}

	////////////////////////////////////////////////////////////////////////////////
	//                             Descriptor Binding                             //
	////////////////////////////////////////////////////////////////////////////////
	void DX12GraphicCommandList::SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE heapType, ID3D12DescriptorHeap* heap)
	{
		if (m_DescriptorHeaps[heapType] != heap)
		{
			m_DescriptorHeaps[heapType] = heap;
			BindDescriptorHeaps();
		}
	}

	void DX12GraphicCommandList::BindDescriptorHeaps()
	{
		UINT numDescriptorHeaps = 0;
		ID3D12DescriptorHeap* descriptorHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES] = {};

		for (uint32_t i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; ++i)
		{
			ID3D12DescriptorHeap* descriptorHeap = m_DescriptorHeaps[i];
			if (descriptorHeap)
			{
				descriptorHeaps[numDescriptorHeaps++] = descriptorHeap;
			}
		}

		m_GraphicCmdList->SetDescriptorHeaps(numDescriptorHeaps, descriptorHeaps);
	}
}