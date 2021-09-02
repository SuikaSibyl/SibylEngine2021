#include "SIByLpch.h"
#include "DX12CommandList.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

#include "Platform/DirectX12/AbstractAPI/Bottom/DX12Resource.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12ResourceStateTracker.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12DynamicDescriptorHeap.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12UploadBuffer.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12FrameResources.h"

namespace SIByL
{
	DX12CommandList::DX12CommandList(D3D12_COMMAND_LIST_TYPE type)
		:m_d3d12CommandListType(type)
	{
		auto device = DX12Context::GetDevice();

		DXCall(device->CreateCommandAllocator(m_d3d12CommandListType, IID_PPV_ARGS(&m_d3d12CommandAllocator)));

		DXCall(device->CreateCommandList(0, m_d3d12CommandListType, m_d3d12CommandAllocator.Get(),
			nullptr, IID_PPV_ARGS(&m_d3d12CommandList)));

		m_UploadBuffer = std::make_unique<DX12UploadBuffer>();

		m_ResourceStateTracker = std::make_unique<DX12ResourceStateTracker>();

		for (int i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; ++i)
		{
			m_DynamicDescriptorHeap[i] = std::make_unique<DX12DynamicDescriptorHeap>(static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(i));
			m_DescriptorHeaps[i] = nullptr;
		}

		m_ResourceStateTracker = CreateScope<DX12ResourceStateTracker>();
	}

	DX12CommandList::~DX12CommandList()
	{

	}

	bool DX12CommandList::Close(DX12CommandList& pendingCommandList)
	{
		// Flush any remaining barriers.
		FlushResourceBarriers();

		m_d3d12CommandList->Close();

		// Flush pending resource barriers.
		uint32_t numPendingBarriers = m_ResourceStateTracker->FlushPendingResourceBarriers(pendingCommandList);
		// Commit the final resource state to the global state.
		m_ResourceStateTracker->CommitFinalResourceStates();

		return numPendingBarriers > 0;
	}

	void DX12CommandList::Close()
	{
		FlushResourceBarriers();
		m_d3d12CommandList->Close();
	}

	void DX12CommandList::Reset()
	{
		DXCall(m_d3d12CommandAllocator->Reset());
		ThrowIfFailed(m_d3d12CommandList->Reset(m_d3d12CommandAllocator.Get(), nullptr));

		m_ResourceStateTracker->Reset();
		m_UploadBuffer->Reset();

		ReleaseTrackedObjects();

		for (int i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; ++i)
		{
			m_DynamicDescriptorHeap[i]->Reset();
			m_DescriptorHeaps[i] = nullptr;
		}

		m_RootSignature = nullptr;
		m_ComputeCommandList = nullptr;
	}

	void DX12CommandList::ReleaseTrackedObjects()
	{
		m_TrackedObjects.clear();
	}

	void DX12CommandList::TransitionBarrier(const DX12Resource& resource, 
		D3D12_RESOURCE_STATES stateAfter, UINT subResource, bool flushBarriers)
	{
		auto d3d12Resource = resource.GetD3D12Resource();
		if (d3d12Resource)
		{
			// The "before" state is not important. It will be resolved by the resource state tracker.
			auto barrier = CD3DX12_RESOURCE_BARRIER::Transition(d3d12Resource.Get(), 
				D3D12_RESOURCE_STATE_COMMON, stateAfter, subResource);
			m_ResourceStateTracker->ResourceBarrier(barrier);
		}

		if (flushBarriers)
		{
			FlushResourceBarriers();
		}
	}

	void DX12CommandList::UAVBarrier(const DX12Resource& resource, bool flushBarriers)
	{
		auto d3d12Resource = resource.GetD3D12Resource();
		auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(d3d12Resource.Get());

		m_ResourceStateTracker->ResourceBarrier(barrier);

		if (flushBarriers)
		{
			FlushResourceBarriers();
		}
	}

	void DX12CommandList::AliasingBarrier(const DX12Resource& beforeResource, 
		const DX12Resource& afterResource, bool flushBarriers)
	{
		auto d3d12BeforeResource = beforeResource.GetD3D12Resource();
		auto d3d12AfterResource = afterResource.GetD3D12Resource();
		auto barrier = CD3DX12_RESOURCE_BARRIER::Aliasing(d3d12BeforeResource.Get(), d3d12AfterResource.Get());

		m_ResourceStateTracker->ResourceBarrier(barrier);

		if (flushBarriers)
		{
			FlushResourceBarriers();
		}
	}

	void DX12CommandList::CopyResource(DX12Resource& dstRes, const DX12Resource& srcRes)
	{
		TransitionBarrier(dstRes, D3D12_RESOURCE_STATE_COPY_DEST);
		TransitionBarrier(srcRes, D3D12_RESOURCE_STATE_COPY_SOURCE);

		FlushResourceBarriers();

		m_d3d12CommandList->CopyResource(dstRes.GetD3D12Resource().Get(), srcRes.GetD3D12Resource().Get());

		TrackResource(dstRes);
		TrackResource(srcRes);
	}

	void DX12CommandList::FlushResourceBarriers()
	{
		m_ResourceStateTracker->FlushResourceBarriers(*this);
	}

	void DX12CommandList::TrackObject(Microsoft::WRL::ComPtr<ID3D12Object> object)
	{
		m_TrackedObjects.push_back(object);
	}

	void DX12CommandList::TrackResource(const DX12Resource& res)
	{
		TrackObject(res.GetD3D12Resource());
	}

	////////////////////////////////////////////////////////////////////////////////
	//                             Descriptor Binding                             //
	////////////////////////////////////////////////////////////////////////////////
	void DX12CommandList::SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE heapType, ComPtr<ID3D12DescriptorHeap> heap)
	{
		if (m_DescriptorHeaps[heapType] != heap)
		{
			m_DescriptorHeaps[heapType] = heap;
			BindDescriptorHeaps();
		}
	}

	void DX12CommandList::BindDescriptorHeaps()
	{
		UINT numDescriptorHeaps = 0;
		ID3D12DescriptorHeap* descriptorHeaps[D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES] = {};

		for (uint32_t i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; ++i)
		{
			ID3D12DescriptorHeap* descriptorHeap = m_DescriptorHeaps[i].Get();
			if (descriptorHeap)
			{
				descriptorHeaps[numDescriptorHeaps++] = descriptorHeap;
			}
		}

		m_d3d12CommandList->SetDescriptorHeaps(numDescriptorHeaps, descriptorHeaps);
	}

	DX12GraphicCommandList::DX12GraphicCommandList()
	{
		DXCall(DX12Context::GetDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_CommandAllocator)));
		DXCall(DX12Context::GetDevice()->CreateCommandList(0,
			D3D12_COMMAND_LIST_TYPE_DIRECT,
			m_CommandAllocator.Get(),
			nullptr,
			IID_PPV_ARGS(&m_GraphicCmdList)));

		m_GraphicCmdList->Close();

		for (uint32_t i = 0; i < D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES; ++i)
		{
			m_DescriptorHeaps[i] = nullptr;
		}
	}

	void DX12GraphicCommandList::Restart()
	{
		ComPtr<ID3D12CommandAllocator> commandAllocator = DX12FrameResourcesManager::GetCurrentAllocator();
		DXCall(commandAllocator->Reset());
		DXCall(m_GraphicCmdList->Reset(commandAllocator.Get(), nullptr));

		m_IsClosed = false;

		// INIT
		BindDescriptorHeaps();
	}

	void DX12GraphicCommandList::Execute()
	{
		DXCall(m_GraphicCmdList->Close());
		ID3D12CommandList* cmdLists[] = { m_GraphicCmdList.Get() };
		DX12Context::GetCommandQueue()->ExecuteCommandLists(_countof(cmdLists), cmdLists);
		m_IsClosed = true;
	}

	////////////////////////////////////////////////////////////////////////////////
	//                             Descriptor Binding                             //
	////////////////////////////////////////////////////////////////////////////////
	void DX12GraphicCommandList::SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE heapType, ComPtr<ID3D12DescriptorHeap> heap)
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
			ID3D12DescriptorHeap* descriptorHeap = m_DescriptorHeaps[i].Get();
			if (descriptorHeap)
			{
				descriptorHeaps[numDescriptorHeaps++] = descriptorHeap;
			}
		}

		m_GraphicCmdList->SetDescriptorHeaps(numDescriptorHeaps, descriptorHeaps);
	}
}