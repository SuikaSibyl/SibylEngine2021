#pragma once
#include "SIByLpch.h"
#include "Sibyl/Graphic/AbstractAPI/Middle/CommandList.h"

namespace SIByL
{
	class DX12Resource;
	class DX12ResourceStateTracker;

	class DX12CommandList
	{
	public:
		/////////////////////////////////////////////////////////
		///				        Constructors     	          ///
		DX12CommandList();

		/////////////////////////////////////////////////////////
		///				        Commands         	          ///

		// Work Submission Commands -----------------------------
		
		bool Close(DX12CommandList& pendingCommandList);
		// Just close the command list. This is useful for pending command lists.
		void Close();

		// Resource Barrier Series ------------------------------

		void TransitionBarrier(const DX12Resource& resource, D3D12_RESOURCE_STATES stateAfter, 
			UINT subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, bool flushBarriers = false);

		void UAVBarrier(const DX12Resource& resource, bool flushBarriers = false);
		
		void AliasingBarrier(const DX12Resource& beforeResource, const DX12Resource& afterResource, bool flushBarriers = false);

		void FlushResourceBarriers();

		// Resource Manipulate Series ------------------------------

		void CopyResource(DX12Resource& dstRes, const DX12Resource& srcRes);
		
		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		ComPtr<ID3D12GraphicsCommandList> GetGraphicsCommandList() const
		{
			return m_d3d12CommandList;
		}

		ComPtr<ID3D12GraphicsCommandList> m_d3d12CommandList;

	private:
		void TrackObject(ComPtr<ID3D12Object> object);
		void TrackResource(const DX12Resource& res);

	private:
		Scope<DX12ResourceStateTracker> m_ResourceStateTracker;

		// Objects that are being tracked by a command list that is "in-flight" on 
		// the command-queue and cannot be deleted. To ensure objects are not deleted 
		// until the command list is finished executing, a reference to the object
		// is stored. The referenced objects are released when the command list is 
		// reset.
		using TrackedObjects = std::vector<ComPtr<ID3D12Object>>;
		TrackedObjects m_TrackedObjects;
	};

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

		inline ComPtr<ID3D12GraphicsCommandList> Get() { return m_GraphicCmdList.Get(); }
		inline ComPtr<ID3D12CommandAllocator> GetAllocator() { return m_CommandAllocator.Get(); }

	private:
		ComPtr<ID3D12GraphicsCommandList>	m_GraphicCmdList;
		ComPtr<ID3D12CommandAllocator>		m_CommandAllocator;

	private:
		bool m_IsClosed = false;

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