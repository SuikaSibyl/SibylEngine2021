#include "SIByLpch.h"
#include "DX12SwapChain.h"
#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/Windows/Window/WindowsWindow.h"
#include "Platform/DirectX12/AbstractAPI/Bottom/DX12DescriptorAllocator.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandList.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandQueue.h"

namespace SIByL
{
    DX12SwapChain::DX12SwapChain()
        :SwapChain(WindowsWindow::Main->GetWidth(), WindowsWindow::Main->GetHeight())
    {
        Ref<DX12CommandQueue> cmdQueue = DX12Context::GetSCommandQueue();
        Ref<DX12CommandList> cmdList = cmdQueue->GetCommandList();
        DX12Context::SetInFlightSCmdList(cmdList);

        CreateSwapChain(WindowsWindow::Main->GetWidth(), WindowsWindow::Main->GetHeight());
        CreateDepthStencil(WindowsWindow::Main->GetWidth(), WindowsWindow::Main->GetHeight());
        SetViewportRect(WindowsWindow::Main->GetWidth(), WindowsWindow::Main->GetHeight());

        RecreateRenderTarget(WindowsWindow::Main->GetWidth(), WindowsWindow::Main->GetHeight());

        cmdQueue->ExecuteCommandList(cmdList);
    }

    DX12SwapChain::DX12SwapChain(int width, int height)
        :SwapChain(width, height)
    {
        Ref<DX12CommandQueue> cmdQueue = DX12Context::GetSCommandQueue();
        Ref<DX12CommandList> cmdList = cmdQueue->GetCommandList();
        DX12Context::SetInFlightSCmdList(cmdList);

        CreateSwapChain(width, height);
        CreateDepthStencil(width, height);
        SetViewportRect(width, height);

        RecreateRenderTarget(WindowsWindow::Main->GetWidth(), WindowsWindow::Main->GetHeight());

        cmdQueue->ExecuteCommandList(cmdList);
    }

    void DX12SwapChain::CreateSwapChain(int width, int height)
    {
        m_SwapChain.Reset();
        // Swap Chain Descriptor
        DXGI_SWAP_CHAIN_DESC swapChainDesc;
        swapChainDesc.BufferDesc.Width = width;
        swapChainDesc.BufferDesc.Height = height;
        swapChainDesc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swapChainDesc.BufferDesc.RefreshRate.Denominator = 1;
        swapChainDesc.BufferDesc.RefreshRate.Numerator = 60;
        swapChainDesc.BufferDesc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
        swapChainDesc.BufferDesc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;
        swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swapChainDesc.OutputWindow = (HWND)WindowsWindow::Main->GetNativeWindow();
        swapChainDesc.SampleDesc.Count = 1;
        swapChainDesc.SampleDesc.Quality = 0;
        swapChainDesc.Windowed = true;
        swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
        swapChainDesc.BufferCount = 2;
        swapChainDesc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;

        DXCall(DX12Context::GetDxgiFactory()->CreateSwapChain(DX12Context::GetCommandQueue(), 
            &swapChainDesc, m_SwapChain.GetAddressOf()));
    }

    void DX12SwapChain::CreateDepthStencil(int width, int height)
    {
        m_DepthStencilResource = CreateRef<DX12DepthStencilResource>(width, height, DX12Context::GetInFlightSCmdList());
    }

    void DX12SwapChain::RecreateRenderTarget(int width, int height)
    {
        ComPtr<ID3D12Resource> m_SwapChainBuffer[2];
        for (int i = 0; i < 2; i++)
        {
            // Get buffer of the swap chain
            m_SwapChain->GetBuffer(i, IID_PPV_ARGS(m_SwapChainBuffer[i].GetAddressOf()));
            m_SwapChainResource[i] = CreateRef<DX12SwapChainResource>(width, height, m_SwapChainBuffer[i]);
        }
    }

    void DX12SwapChain::SetRenderTarget()
    {
        Ref<DX12CommandList> sCmdList = DX12Context::GetInFlightSCmdList();
        ID3D12GraphicsCommandList* cmdList = DX12Context::GetInFlightDXGraphicCommandList();

        sCmdList->TransitionBarrier(
            *m_SwapChainResource[m_CurrentBackBuffer]->GetResource(), 
            D3D12_RESOURCE_STATE_RENDER_TARGET);

        cmdList->RSSetViewports(1, &viewPort);
        cmdList->RSSetScissorRects(1, &scissorRect);

        // Clear Render Targets
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_SwapChainResource[m_CurrentBackBuffer]->GetRTVCpuHandle();
        cmdList->ClearRenderTargetView(rtvHandle, DirectX::Colors::Transparent, 0, nullptr);

        D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = m_DepthStencilResource->GetDSVCpuHandle();
        cmdList->ClearDepthStencilView(dsvHandle,	//DSV描述符句柄
            D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
            1.0f,	//默认深度值
            0,	//默认模板值
            0,	//裁剪矩形数量
            nullptr);	//裁剪矩形指针

        cmdList->OMSetRenderTargets(1,//待绑定的RTV数量
            &rtvHandle,	//指向RTV数组的指针
            true,	//RTV对象在堆内存中是连续存放的
            &dsvHandle);	//指向DSV的指针
    }

    void DX12SwapChain::PreparePresent()
    {
        Ref<DX12CommandList> sCmdList = DX12Context::GetInFlightSCmdList();

        sCmdList->TransitionBarrier(
            *m_SwapChainResource[m_CurrentBackBuffer]->GetResource(),
            D3D12_RESOURCE_STATE_PRESENT);
    }

    void DX12SwapChain::Present()
    {
        DXCall(m_SwapChain->Present(0, 0));
        m_CurrentBackBuffer = (m_CurrentBackBuffer + 1) % 2;
    }

    void DX12SwapChain::Reisze(uint32_t width, uint32_t height)
    {
        if (width <= 0 || height <= 0)
        {
            return;
        }

        Ref<DX12CommandQueue> cmdQueue = DX12Context::GetSCommandQueue();
        Ref<DX12CommandList> cmdList = cmdQueue->GetCommandList();
        DX12Context::SetInFlightSCmdList(cmdList);

        uint64_t fence = cmdQueue->Signal();
        cmdQueue->WaitForFenceValue(fence);

        for (UINT i = 0; i < 2; i++)
            m_SwapChainResource[i]->Reset();
        
        m_SwapChain->ResizeBuffers(
            2, // Swap chain number
            width, height, 
            DXGI_FORMAT_R8G8B8A8_UNORM,
            DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH);

        RecreateRenderTarget(width ,height);
        RecreateDepthStencil(width, height);
        SetViewportRect(width, height);
        m_CurrentBackBuffer = 0;

        cmdQueue->ExecuteCommandList(cmdList);

    }
    
    void DX12SwapChain::RecreateDepthStencil(int width, int height)
    {
        m_DepthStencilResource->Resize(width, height, DX12Context::GetInFlightSCmdList());
    }

    void DX12SwapChain::SetViewportRect(int width, int height)
    {
        // Set Viewport
        viewPort.TopLeftX = 0;
        viewPort.TopLeftY = 0;
        viewPort.Width =(float)width;
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