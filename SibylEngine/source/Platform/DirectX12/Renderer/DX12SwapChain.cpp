#include "SIByLpch.h"
#include "DX12SwapChain.h"
#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/Windows/Window/WindowsWindow.h"
#include "Platform/DirectX12/Core/DescriptorAllocator.h"

namespace SIByL
{
    DX12SwapChain::DX12SwapChain()
        :SwapChain(WindowsWindow::Main->GetWidth(), WindowsWindow::Main->GetHeight())
    {
        CreateSwapChain(WindowsWindow::Main->GetWidth(), WindowsWindow::Main->GetHeight());
    }

    DX12SwapChain::DX12SwapChain(int width, int height)
        :SwapChain(width, height)
    {
        CreateSwapChain(width, height);
        SetViewportRect(width, height);
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

    void DX12SwapChain::BindRenderTarget()
    {
        // Allocate descriptors from Heap
        m_DescriptorAllocation = DX12Context::GetRtvDescriptorAllocator()->Allocate(2u);
        CD3DX12_CPU_DESCRIPTOR_HANDLE  handle(m_DescriptorAllocation.GetDescriptorHandle());
        for (int i = 0; i < 2; i++)
        {
            // Get buffer of the swap chain
            m_SwapChain->GetBuffer(i, IID_PPV_ARGS(m_SwapChainBuffer[i].GetAddressOf()));
            // Create RTV in heap
            DX12Context::GetDevice()->CreateRenderTargetView(
                m_SwapChainBuffer[i].Get(),
                nullptr,
                handle);
            // Offset the handle to the next rtv
            handle.Offset(1, DX12Context::GetRtvDescriptorSize());
        }
    }

    void DX12SwapChain::SetRenderTarget()
    {
        ID3D12GraphicsCommandList* cmdList = DX12Context::GetDXGraphicCommandList();
        cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
            m_SwapChainBuffer[m_CurrentBackBuffer].Get(),
            D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));

        cmdList->RSSetViewports(1, &viewPort);
        cmdList->RSSetScissorRects(1, &scissorRect);

        // Clear Render Targets
        D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = m_DescriptorAllocation.GetDescriptorHandle(m_CurrentBackBuffer);
        cmdList->ClearRenderTargetView(rtvHandle, DirectX::Colors::Black, 0, nullptr);
        //D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        //cmdList->ClearDepthStencilView(dsvHandle,	//DSV描述符句柄
        //    D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL,	//FLAG
        //    1.0f,	//默认深度值
        //    0,	//默认模板值
        //    0,	//裁剪矩形数量
        //    nullptr);	//裁剪矩形指针

        cmdList->OMSetRenderTargets(1,//待绑定的RTV数量
            &rtvHandle,	//指向RTV数组的指针
            true,	//RTV对象在堆内存中是连续存放的
            nullptr);	//指向DSV的指针
    }

    void DX12SwapChain::PreparePresent()
    {
        ID3D12GraphicsCommandList* cmdList = DX12Context::GetDXGraphicCommandList();
        cmdList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::Transition(
            m_SwapChainBuffer[m_CurrentBackBuffer].Get(),
            D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT));//从渲染目标到呈现
    }

    void DX12SwapChain::Present()
    {
        DXCall(m_SwapChain->Present(0, 0));
        m_CurrentBackBuffer = (m_CurrentBackBuffer + 1) % 2;
    }

    void DX12SwapChain::SetViewportRect(int width, int height)
    {
        // Set Viewport
        viewPort.TopLeftX = 0;
        viewPort.TopLeftY = 0;
        viewPort.Width = width;
        viewPort.Height = height;
        viewPort.MaxDepth = 1.0f;
        viewPort.MinDepth = 0.0f;

        // Set Scissor Rect
        scissorRect.left = 0;
        scissorRect.top = 0;
        scissorRect.right = width;
        scissorRect.bottom = height;
    }
}