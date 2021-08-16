#include "SIByLpch.h"
#include "DX12SwapChain.h"
#include "Platform/DirectX12/Common/DX12Utility.h"
#include "Platform/Windows/WindowsWindow.h"

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
}