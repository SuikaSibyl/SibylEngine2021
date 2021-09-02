#include "SIByLpch.h"
#include "DX12FrameBuffer.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

#include "Platform/DirectX12/AbstractAPI/Bottom/DX12CommonResource.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12Texture.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandList.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandQueue.h"

namespace SIByL
{
	///////////////////////////////////////////////////////////////////////////
	///                      Constructors / Destructors                     ///
	///////////////////////////////////////////////////////////////////////////
	
	// Create From Descriptor
	// ----------------------------------------------------------
	DX12FrameBuffer::DX12FrameBuffer(const FrameBufferDesc& desc)
		:m_Desc(desc)
	{
		m_RenderTargetResource = CreateRef<DX12RenderTargetResource>
			(m_Desc.Width, m_Desc.Height, DX12Context::GetInFlightSCmdList());

		m_DepthStencilResource = CreateRef<DX12DepthStencilResource>
			(m_Desc.Width, m_Desc.Height, DX12Context::GetInFlightSCmdList());
	}

	DX12FrameBuffer::~DX12FrameBuffer()
	{

	}

	///////////////////////////////////////////////////////////////////////////
	///						        Manipulator		                        ///
	///////////////////////////////////////////////////////////////////////////
	void DX12FrameBuffer::Bind()
	{
		Ref<DX12CommandList> cmdList = DX12Context::GetInFlightSCmdList();

		cmdList->GetGraphicsCommandList()->ResourceBarrier(1,
			&CD3DX12_RESOURCE_BARRIER::Transition(m_RenderTargetResource->GetResource()->GetD3D12Resource().Get(),//转换资源为后台缓冲区资源
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换

		cmdList->GetGraphicsCommandList()->OMSetRenderTargets(1,
			&m_RenderTargetResource->GetRTVCpuHandle(),
			true,
			&m_DepthStencilResource->GetDSVCpuHandle());
	}

	void DX12FrameBuffer::Unbind()
	{
		SwapChain* swapChain = DX12Context::GetSwapChain();
		swapChain->SetRenderTarget();

		//
		Ref<DX12CommandList> cmdList = DX12Context::GetInFlightSCmdList();
		cmdList->TransitionBarrier(*m_RenderTargetResource->GetResource(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
		cmdList->GetGraphicsCommandList()->ResourceBarrier(1, 
			&CD3DX12_RESOURCE_BARRIER::Transition(m_RenderTargetResource->GetResource()->GetD3D12Resource().Get(),//转换资源为后台缓冲区资源
				D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));//从呈现到渲染目标转换

	}

	void DX12FrameBuffer::Resize(uint32_t width, uint32_t height)
	{
		if (width <= 0 || height <= 0)
		{
			return;
		}

		m_Desc.Width = width;
		m_Desc.Height = height;

		Ref<DX12CommandQueue> cmdQueue = DX12Context::GetSCommandQueue();
		Ref<DX12CommandList> cmdList = cmdQueue->GetCommandList();
		DX12Context::SetInFlightSCmdList(cmdList);

		m_RenderTargetResource->Resize(width, height, cmdList);
		m_DepthStencilResource->Resize(width, height, cmdList);

		cmdQueue->ExecuteCommandList(cmdList);
	}

	void DX12FrameBuffer::ClearBuffer()
	{

	}

	void DX12FrameBuffer::ClearRgba()
	{

	}

	void DX12FrameBuffer::ClearDepthStencil()
	{

	}

	void* DX12FrameBuffer::GetColorAttachment()
	{
		D3D12_GPU_DESCRIPTOR_HANDLE* handle = &m_RenderTargetResource->GetImGuiGpuHandle();
		return *(void**)(handle);
	}

	///////////////////////////////////////////////////////////////////////////
	///                                Caster                               ///
	///////////////////////////////////////////////////////////////////////////

	Ref<Texture2D> DX12FrameBuffer::ColorAsTexutre()
	{
		Ref<Texture2D> texture = nullptr;
		//texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width,
		//	m_Desc.Height, m_Desc.Channel, Texture2D::Format::R24G8));
		return texture;
	}

	Ref<Texture2D> DX12FrameBuffer::DepthStencilAsTexutre()
	{
		Ref<Texture2D> texture = nullptr;
		//texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width,
		//	m_Desc.Height, m_Desc.Channel, Texture2D::Format::R24G8));
		return texture;
	}
}