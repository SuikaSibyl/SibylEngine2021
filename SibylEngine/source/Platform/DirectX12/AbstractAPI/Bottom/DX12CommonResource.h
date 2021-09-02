#pragma once

#include "Platform/DirectX12/AbstractAPI/Bottom/DX12Resource.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12DescriptorAllocation.h"
#include "Platform/DirectX12/ImGui/ImGuiLayerDX12.h"

namespace SIByL
{
	class DX12CommandList;

	///////////////////////////////////////////////////////////////////////////////////
	//																				 //
	//						    	Swap Chain Resource								 //
	//																				 //
	///////////////////////////////////////////////////////////////////////////////////

	class DX12SwapChainResource
	{
	public:
		DX12SwapChainResource(const uint32_t& width, const uint32_t& height, ComPtr<ID3D12Resource> resource);

		/////////////////////////////////////////////////////////
		///				        Manipulator     	          ///
		void Reset();
		void Resize(const uint32_t& width, const uint32_t& height, ComPtr<ID3D12Resource> resource);

		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		DX12Resource* GetResource() { return m_Resource.get(); }
		D3D12_CPU_DESCRIPTOR_HANDLE GetRTVCpuHandle();

	protected:
		/////////////////////////////////////////////////////////
		///				      Local Function     	          ///
		void CreateResource(ComPtr<ID3D12Resource> resource);

	protected:
		Scope<DX12Resource> m_Resource;
		uint32_t m_Width, m_Height;
		DX12DescriptorAllocation m_RTVDescriptorAllocation;
	};



	///////////////////////////////////////////////////////////////////////////////////
	//																				 //
	//							Depth Stencil Resource								 //
	//																				 //
	///////////////////////////////////////////////////////////////////////////////////

	class DX12DepthStencilResource
	{
	public:
		DX12DepthStencilResource(const uint32_t& width, const uint32_t& height, Ref<DX12CommandList> pSCommandList);

		/////////////////////////////////////////////////////////
		///				        Manipulator     	          ///
		void Reset();
		void Resize(const uint32_t& width, const uint32_t& height, Ref<DX12CommandList> pSCommandList);
		
		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		DX12Resource* GetResource() { return m_Resource.get(); }
		D3D12_CPU_DESCRIPTOR_HANDLE GetDSVCpuHandle();

	protected:
		/////////////////////////////////////////////////////////
		///				      Local Function     	          ///
		void CreateResource(Ref<DX12CommandList> pSCommandList);

	protected:
		Scope<DX12Resource> m_Resource;
		uint32_t m_Width, m_Height;
		DX12DescriptorAllocation m_DSVDescriptorAllocation;
	};



	///////////////////////////////////////////////////////////////////////////////////
	//																				 //
	//							Render Target Resource								 //
	//																				 //
	///////////////////////////////////////////////////////////////////////////////////

	class DX12RenderTargetResource
	{
	public:
		DX12RenderTargetResource(const uint32_t& width, const uint32_t& height, Ref<DX12CommandList> pSCommandList);

		/////////////////////////////////////////////////////////
		///				        Manipulator     	          ///
		void Reset();
		void Resize(const uint32_t& width, const uint32_t& height, Ref<DX12CommandList> pSCommandList);

		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		DX12Resource* GetResource() { return m_Resource.get(); }
		D3D12_CPU_DESCRIPTOR_HANDLE GetRTVCpuHandle();
		D3D12_CPU_DESCRIPTOR_HANDLE GetSRVCpuHandle();
		D3D12_GPU_DESCRIPTOR_HANDLE GetSRVGpuHandle();
		D3D12_GPU_DESCRIPTOR_HANDLE GetImGuiGpuHandle();

	protected:
		/////////////////////////////////////////////////////////
		///				      Local Function     	          ///
		void CreateResource(Ref<DX12CommandList> pSCommandList);

	protected:
		Scope<DX12Resource> m_Resource;
		uint32_t m_Width, m_Height;
		DX12DescriptorAllocation m_RTVDescriptorAllocation;
		DX12DescriptorAllocation m_SRVDescriptorAllocationCpu;
		DX12DescriptorAllocation m_SRVDescriptorAllocationGpu;
		ImGuiLayerDX12::ImGuiAllocation m_ImGuiAllocation;
	};
}