#pragma once

#include "SIByLpch.h"
#include "Sibyl/Basic/ThreadSafeQueue.h"

namespace SIByL
{
    class DX12CommandList;

	class DX12CommandQueue
	{
	public:
		DX12CommandQueue(D3D12_COMMAND_LIST_TYPE type);
		virtual ~DX12CommandQueue();

		// Get an available command list from the command queue.
		Ref<DX12CommandList> GetCommandList();

		// Execute a command list.
		// Returns the fence value to wait for for this command list.
		uint64_t ExecuteCommandList(Ref<DX12CommandList> commandList);
		uint64_t ExecuteCommandLists(const std::vector<Ref<DX12CommandList> >& commandLists);

		uint64_t Signal();
		bool IsFenceComplete(uint64_t fenceValue);
		void WaitForFenceValue(uint64_t fenceValue);
		void Flush();
        // Wait for another command queue to finish.
        void Wait(const DX12CommandQueue& other);

        Microsoft::WRL::ComPtr<ID3D12CommandQueue> GetD3D12CommandQueue() const;

    private:
        // Free any command lists that are finished processing on the command queue.
        void ProccessInFlightCommandLists();

        // Keep track of command allocators that are "in-flight"
        // The first member is the fence value to wait for, the second is the 
        // a shared pointer to the "in-flight" command list.
        using CommandListEntry = std::tuple<uint64_t, Ref<DX12CommandList> >;

        D3D12_COMMAND_LIST_TYPE                         m_CommandListType;
        Microsoft::WRL::ComPtr<ID3D12CommandQueue>      m_d3d12CommandQueue;
        Microsoft::WRL::ComPtr<ID3D12Fence>             m_d3d12Fence;
        std::atomic_uint64_t                            m_FenceValue;

        ThreadSafeQueue<CommandListEntry>               m_InFlightCommandLists;
        ThreadSafeQueue<Ref<DX12CommandList>>           m_AvailableCommandLists;

        // A thread to process in-flight command lists.
        std::thread m_ProcessInFlightCommandListsThread;
        std::atomic_bool m_bProcessInFlightCommandLists;
        std::mutex m_ProcessInFlightCommandListsThreadMutex;
        std::condition_variable m_ProcessInFlightCommandListsThreadCV;
    };
}