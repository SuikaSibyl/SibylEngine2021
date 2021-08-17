#include "SIByLpch.h"
#include "DX12CommandList.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

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
		DXCall(m_CommandAllocator->Reset());
		DXCall(m_GraphicCmdList->Reset(m_CommandAllocator.Get(), nullptr));
	}

	void DX12GraphicCommandList::Execute()
	{
		DXCall(m_GraphicCmdList->Close());
		ID3D12CommandList* cmdLists[] = { m_GraphicCmdList.Get() };
		DX12Context::GetCommandQueue()->ExecuteCommandLists(_countof(cmdLists), cmdLists);
	}
}