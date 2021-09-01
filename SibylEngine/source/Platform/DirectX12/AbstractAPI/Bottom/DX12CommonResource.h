#pragma once

#include "Platform/DirectX12/AbstractAPI/Bottom/DX12Resource.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12DescriptorAllocation.h"

namespace SIByL
{
	class DX12DepthStencilResource
	{
	public:
		DX12DepthStencilResource(const uint32_t& width, const uint32_t& height);


		/////////////////////////////////////////////////////////
		///				        Manipulator     	          ///
		void Reset();
		void Resize(const uint32_t& width, const uint32_t& height);
		
		/////////////////////////////////////////////////////////
		///				     Fetcher / Setter		          ///
		DX12Resource* GetResource() { return m_Resource.get(); }

	protected:
		/////////////////////////////////////////////////////////
		///				      Local Function     	          ///
		void CreateResource();

	protected:
		Scope<DX12Resource> m_Resource;
		uint32_t m_Width, m_Height;
		DescriptorAllocation m_DescriptorAllocation;
	};
}