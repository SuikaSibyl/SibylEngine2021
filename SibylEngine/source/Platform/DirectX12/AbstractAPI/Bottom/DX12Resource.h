#pragma once

#include "SIByLpch.h"

namespace SIByL
{
	class DX12Resource
	{
	public:
		/////////////////////////////////////////////////////////
		///				    	Constructors		          ///
		/////////////////////////////////////////////////////////
		DX12Resource(const std::wstring& name = L"");
		DX12Resource(const D3D12_RESOURCE_DESC& resourceDesc,
			const D3D12_CLEAR_VALUE* clearValue = nullptr,
			const std::wstring& name = L"");
		DX12Resource(ComPtr<ID3D12Resource> resource, const std::wstring& name = L"");
		DX12Resource(const DX12Resource& copy);
		DX12Resource(DX12Resource&& copy);

		DX12Resource& operator=(const DX12Resource& other);
		DX12Resource& operator=(DX12Resource&& other);

		virtual ~DX12Resource();

		/////////////////////////////////////////////////////////
		///				    	Functions   		          ///
		/////////////////////////////////////////////////////////
		/**
		 * Check to see if the underlying resource is valid.
		 */
		bool IsValid() const
		{
			return (m_d3d12Resource != nullptr);
		}

		// Get access to the underlying D3D12 resource
		ComPtr<ID3D12Resource> GetD3D12Resource() const
		{
			return m_d3d12Resource;
		}

		D3D12_RESOURCE_DESC GetD3D12ResourceDesc() const
		{
			D3D12_RESOURCE_DESC resDesc = {};
			if (m_d3d12Resource)
			{
				resDesc = m_d3d12Resource->GetDesc();
			}

			return resDesc;
		}

		// Replace the D3D12 resource
		// Should only be called by the CommandList.
		void SetD3D12Resource(ComPtr<ID3D12Resource> d3d12Resource, const D3D12_CLEAR_VALUE* clearValue);

		/**
		 * Set the name of the resource. Useful for debugging purposes.
		 * The name of the resource will persist if the underlying D3D12 resource is
		 * replaced with SetD3D12Resource.
		 */
		void SetName(const std::wstring& name);

		/**
		 * Release the underlying resource.
		 * This is useful for swap chain resizing.
		 */
		virtual void Reset();

	protected:
		/////////////////////////////////////////////////////////
		///				    	Datas      		              ///
		/////////////////////////////////////////////////////////
		// The underlying D3D12 resource.
		ComPtr<ID3D12Resource> m_d3d12Resource;
		std::unique_ptr<D3D12_CLEAR_VALUE> m_d3d12ClearValue;
		std::wstring m_ResourceName;
	};
}