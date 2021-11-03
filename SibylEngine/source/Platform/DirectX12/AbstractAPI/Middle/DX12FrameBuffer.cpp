#include "SIByLpch.h"
#include "DX12FrameBuffer.h"

#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"

#include "Platform/DirectX12/AbstractAPI/Middle/DX12FrameBufferTexture.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12CommonResource.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12Texture.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandList.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandQueue.h"

namespace SIByL
{
//	///////////////////////////////////////////////////////////////////////////
//	///                      Constructors / Destructors                     ///
//	///////////////////////////////////////////////////////////////////////////
//	
//	// Create From Descriptor
//	// ----------------------------------------------------------
//	DX12FrameBuffer_v1::DX12FrameBuffer_v1(const FrameBufferDesc_v1& desc)
//		:m_Desc(desc)
//	{
//		m_RenderTargetResource = CreateRef<DX12RenderTargetResource>
//			(m_Desc.Width, m_Desc.Height, DX12Context::GetInFlightSCmdList());
//
//		m_DepthStencilResource = CreateRef<DX12DepthStencilResource>
//			(m_Desc.Width, m_Desc.Height, DX12Context::GetInFlightSCmdList());
//
//		SetViewportRect(m_Desc.Width, m_Desc.Height);
//	}
//
//	DX12FrameBuffer_v1::~DX12FrameBuffer_v1()
//	{
//
//	}
//
//	///////////////////////////////////////////////////////////////////////////
//	///						        Manipulator		                        ///
//	///////////////////////////////////////////////////////////////////////////
//	void DX12FrameBuffer_v1::Bind()
//	{
//		Ref<DX12CommandList> cmdList = DX12Context::GetInFlightSCmdList();
//
//		cmdList->GetGraphicsCommandList()->ResourceBarrier(1,
//			&CD3DX12_RESOURCE_BARRIER::Transition(m_RenderTargetResource->GetResource()->GetD3D12Resource().Get(),//转换资源为后台缓冲区资源
//				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET));//从呈现到渲染目标转换
//
//		cmdList->GetGraphicsCommandList()->RSSetViewports(1, &viewPort);
//		cmdList->GetGraphicsCommandList()->RSSetScissorRects(1, &scissorRect);
//
//		cmdList->GetGraphicsCommandList()->OMSetRenderTargets(1,
//			&m_RenderTargetResource->GetRTVCpuHandle(),
//			true,
//			&m_DepthStencilResource->GetDSVCpuHandle());
//	}
//
//	void DX12FrameBuffer_v1::Unbind()
//	{
//		SwapChain* swapChain = DX12Context::GetSwapChain();
//		swapChain->SetRenderTarget();
//
//		//
//		Ref<DX12CommandList> cmdList = DX12Context::GetInFlightSCmdList();
//		cmdList->TransitionBarrier(*m_RenderTargetResource->GetResource(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
//		cmdList->GetGraphicsCommandList()->ResourceBarrier(1, 
//			&CD3DX12_RESOURCE_BARRIER::Transition(m_RenderTargetResource->GetResource()->GetD3D12Resource().Get(),//转换资源为后台缓冲区资源
//				D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));//从呈现到渲染目标转换
//
//	}
//
//	void DX12FrameBuffer_v1::Resize(uint32_t width, uint32_t height)
//	{
//		if (width <= 0 || height <= 0)
//		{
//			return;
//		}
//
//		m_Desc.Width = width;
//		m_Desc.Height = height;
//
//		SetViewportRect(m_Desc.Width, m_Desc.Height);
//
//		Ref<DX12CommandQueue> cmdQueue = DX12Context::GetSCommandQueue();
//		Ref<DX12CommandList> cmdList = cmdQueue->GetCommandList();
//		DX12Context::SetInFlightSCmdList(cmdList);
//
//		uint64_t fence = cmdQueue->Signal();
//		cmdQueue->WaitForFenceValue(fence);
//
//		m_RenderTargetResource->Resize(width, height, cmdList);
//		m_DepthStencilResource->Resize(width, height, cmdList);
//
//		cmdQueue->ExecuteCommandList(cmdList);
//	}
//
//	void DX12FrameBuffer_v1::ClearBuffer()
//	{
//		Ref<DX12CommandList> sCmdList = DX12Context::GetInFlightSCmdList();
//		ID3D12GraphicsCommandList* cmdList = DX12Context::GetInFlightDXGraphicCommandList();
//
//		// Clear Render Targets
//		D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_RenderTargetResource->GetRTVCpuHandle();
//		cmdList->ClearRenderTargetView(rtvHandle, DirectX::Colors::Transparent, 0, nullptr);
//
//		D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_DepthStencilResource->GetDSVCpuHandle();
//		cmdList->ClearDepthStencilView(dsvHandle,	//DSV描述符句柄
//			D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
//			1.0f,
//			0,
//			0,
//			nullptr);
//
//	}
//
//	void DX12FrameBuffer_v1::ClearRgba()
//	{
//
//	}
//
//	void DX12FrameBuffer_v1::ClearDepthStencil()
//	{
//
//	}
//
//	void* DX12FrameBuffer_v1::GetColorAttachment()
//	{
//		D3D12_GPU_DESCRIPTOR_HANDLE* handle = &m_RenderTargetResource->GetImGuiGpuHandle();
//		return *(void**)(handle);
//	}
//
//	///////////////////////////////////////////////////////////////////////////
//	///                                Caster                               ///
//	///////////////////////////////////////////////////////////////////////////
//
//	Ref<Texture2D> DX12FrameBuffer_v1::ColorAsTexutre()
//	{
//		Ref<Texture2D> texture = nullptr;
//		//texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width,
//		//	m_Desc.Height, m_Desc.Channel, Texture2D::Format::R24G8));
//		return texture;
//	}
//
//	Ref<Texture2D> DX12FrameBuffer_v1::DepthStencilAsTexutre()
//	{
//		Ref<Texture2D> texture = nullptr;
//		//texture.reset(new OpenGLTexture2D(m_TextureObject, m_Desc.Width,
//		//	m_Desc.Height, m_Desc.Channel, Texture2D::Format::R24G8));
//		return texture;
//	}
//
//	void DX12FrameBuffer_v1::SetViewportRect(int width, int height)
//	{
//		// Set Viewport
//		viewPort.TopLeftX = 0;
//		viewPort.TopLeftY = 0;
//		viewPort.Width = (float)width;
//		viewPort.Height = (float)height;
//		viewPort.MaxDepth = 1.0f;
//		viewPort.MinDepth = 0.0f;
//
//		// Set Scissor Rect
//		scissorRect.left = 0;
//		scissorRect.top = 0;
//		scissorRect.right = width;
//		scissorRect.bottom = height;
//	}
//
//	////////////////////////////////////////////////////
//	//					CUDA Interface				  //
//	////////////////////////////////////////////////////
//
//	Ref<PtrCudaTexture> DX12FrameBuffer_v1::GetPtrCudaTexture()
//	{
//		if (mPtrCudaTexture == nullptr)
//		{
//
//		}
//
//		return mPtrCudaTexture;
//	}
//
//	Ref<PtrCudaSurface> DX12FrameBuffer_v1::GetPtrCudaSurface()
//	{
//		if (mPtrCudaSurface == nullptr)
//		{
//#ifdef SIBYL_PLATFORM_CUDA
//			//mPtrCudaSurface = CreateRef<PtrCudaSurface>();
//			//mPtrCudaSurface->RegisterByOpenGLTexture(m_TextureObject, m_Desc.Width, m_Desc.Height);
//#endif // SIBYL_PLATFORM_CUDA
//		}
//
//		return mPtrCudaSurface;
//	}
//
//	void DX12FrameBuffer_v1::ResizePtrCudaTexuture()
//	{
//
//	}
//
//	void DX12FrameBuffer_v1::ResizePtrCudaSurface()
//	{
//		if (mPtrCudaSurface != nullptr)
//		{
//#ifdef SIBYL_PLATFORM_CUDA
//			//mPtrCudaSurface->RegisterByOpenGLTexture(m_TextureObject, m_Desc.Width, m_Desc.Height);
//#endif // SIBYL_PLATFORM_CUDA
//		}
//	}
//
//	void DX12FrameBuffer_v1::CreatePtrCudaTexutre()
//	{
//		if (mPtrCudaTexture != nullptr)
//		{
//
//		}
//	}
//
//	void DX12FrameBuffer_v1::CreatePtrCudaSurface()
//	{
//
//	}

	////////////////////////////////////////////////////
	//					DX12FrameBuffer				  //
	////////////////////////////////////////////////////

	DX12FrameBuffer::DX12FrameBuffer(const FrameBufferDesc& desc)
		:Width(desc.Width), Height(desc.Height)
	{
		ScaleX = desc.ScaleX;
		ScaleY = desc.ScaleY;

		for (FrameBufferTextureFormat format : desc.Formats.Formats)
		{
			if (IsDepthFormat(format))
			{
				DepthStencil = CreateRef<DX12StencilDepth>(FrameBufferTextureDesc{ format,desc.Width,desc.Height });
			}
			else
			{
				RenderTargets.push_back(CreateRef<DX12RenderTarget>(FrameBufferTextureDesc{ format,desc.Width,desc.Height }));
			}
		}
	}

	void DX12FrameBuffer::Bind()
	{
		Ref<DX12CommandList> cmdList = DX12Context::GetInFlightSCmdList();

		cmdList->GetGraphicsCommandList()->RSSetViewports(1, &viewPort);
		cmdList->GetGraphicsCommandList()->RSSetScissorRects(1, &scissorRect);

		D3D12_CPU_DESCRIPTOR_HANDLE* targets = new D3D12_CPU_DESCRIPTOR_HANDLE[RenderTargets.size()];
		for (int i = 0; i < RenderTargets.size(); i++)
		{
			cmdList->GetGraphicsCommandList()->ResourceBarrier(1,
				&CD3DX12_RESOURCE_BARRIER::Transition(RenderTargets[i]->GetResource()->GetD3D12Resource().Get(),
				D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET));
			targets[i] = RenderTargets[i]->m_RenderTargetResource->GetRTVCpuHandle();
		}

		cmdList->GetGraphicsCommandList()->OMSetRenderTargets(RenderTargets.size(),
			targets,
			true,
			&DepthStencil->m_DepthStencilResource->GetDSVCpuHandle());

		delete[] targets;
	}

	void DX12FrameBuffer::CustomViewport(unsigned int xmin, unsigned int xmax, unsigned int ymin, unsigned int ymax)
	{	
		// Set Viewport
		viewPort.TopLeftX = xmin;
		viewPort.TopLeftY = ymin;
		viewPort.Width = (float)xmax;
		viewPort.Height = (float)ymax;
		viewPort.MaxDepth = 1.0f;
		viewPort.MinDepth = 0.0f;

		// Set Scissor Rect
		scissorRect.left = xmin;
		scissorRect.top = ymin;
		scissorRect.right = xmax;
		scissorRect.bottom = ymax;

		Ref<DX12CommandList> cmdList = DX12Context::GetInFlightSCmdList();

		cmdList->GetGraphicsCommandList()->RSSetViewports(1, &viewPort);
		cmdList->GetGraphicsCommandList()->RSSetScissorRects(1, &scissorRect);
	}

	void DX12FrameBuffer::Unbind()
	{
		SwapChain* swapChain = DX12Context::GetSwapChain();
		swapChain->SetRenderTarget();

		//
		Ref<DX12CommandList> cmdList = DX12Context::GetInFlightSCmdList();

		for (int i = 0; i < RenderTargets.size(); i++)
		{
			cmdList->TransitionBarrier(*RenderTargets[i]->GetResource(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
			cmdList->GetGraphicsCommandList()->ResourceBarrier(1,
				&CD3DX12_RESOURCE_BARRIER::Transition(RenderTargets[i]->GetResource()->GetD3D12Resource().Get(),//转换资源为后台缓冲区资源
					D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));//从呈现到渲染目标转换
		}
	}

	void DX12FrameBuffer::ClearBuffer()
	{
		Ref<DX12CommandList> sCmdList = DX12Context::GetInFlightSCmdList();
		ID3D12GraphicsCommandList* cmdList = DX12Context::GetInFlightDXGraphicCommandList();

		for (int i = 0; i < RenderTargets.size(); i++)
		{
			// Clear Render Targets
			D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = RenderTargets[i]->m_RenderTargetResource->GetRTVCpuHandle();
			cmdList->ClearRenderTargetView(rtvHandle, DirectX::Colors::Transparent, 0, nullptr);
		}

		D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = DepthStencil->m_DepthStencilResource->GetDSVCpuHandle();
		cmdList->ClearDepthStencilView(dsvHandle,
			D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
			1.0f,
			0,
			0,
			nullptr);
	}

	unsigned int DX12FrameBuffer::CountColorAttachment()
	{
		return RenderTargets.size();
	}

	void DX12FrameBuffer::Resize(uint32_t width, uint32_t height)
	{
		if (width <= 0 || height <= 0)
		{
			return;
		}

		Width = width;
		Height = height;

		SetViewportRect(Width, Height);

		Ref<DX12CommandQueue> cmdQueue = DX12Context::GetSCommandQueue();
		Ref<DX12CommandList> cmdList = cmdQueue->GetCommandList();
		DX12Context::SetInFlightSCmdList(cmdList);

		uint64_t fence = cmdQueue->Signal();
		cmdQueue->WaitForFenceValue(fence);

		for (int i = 0; i < RenderTargets.size(); i++)
		{
			RenderTargets[i]->Resize(Width, Height);
		}
		DepthStencil->Resize(Width, Height);

		cmdQueue->ExecuteCommandList(cmdList);
	}

	void* DX12FrameBuffer::GetColorAttachment(unsigned int index)
	{
		D3D12_GPU_DESCRIPTOR_HANDLE* handle = &RenderTargets[index]->m_RenderTargetResource->GetImGuiGpuHandle();
		return *(void**)(handle);
	}

	void* DX12FrameBuffer::GetDepthStencilAttachment()
	{
		return nullptr;
	}

	RenderTarget* DX12FrameBuffer::GetRenderTarget(unsigned int index)
	{
		return (RenderTarget*)(RenderTargets[index].get());
	}

	void DX12FrameBuffer::SetViewportRect(int width, int height)
	{
		// Set Viewport
		viewPort.TopLeftX = 0;
		viewPort.TopLeftY = 0;
		viewPort.Width = (float)width;
		viewPort.Height = (float)height;
		viewPort.MaxDepth = 1.0f;
		viewPort.MinDepth = 0.0f;

		// Set Scissor Rect
		scissorRect.left = 0;
		scissorRect.top = 0;
		scissorRect.right = width;
		scissorRect.bottom = height;
	}
}