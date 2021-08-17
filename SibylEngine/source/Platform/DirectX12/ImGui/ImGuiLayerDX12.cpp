#include "SIByLpch.h"
#include "ImGuiLayerDX12.h"

#include "Platform/Windows/WindowsWindow.h"
#include "Platform/Windows/ImGuiWin32Renderer.h"
#include "Platform/DirectX12/ImGui/ImGuiDX12Renderer.h"

namespace SIByL
{
#ifdef RENDER_API_DX12
	ImGuiLayer* ImGuiLayer::Create()
	{
		return new ImGuiLayerDX12();
	}
#endif
	static int const NUM_FRAMES_IN_FLIGHT = 3;

	void ImGuiLayerDX12::PlatformInit()
	{
		g_pd3dSrvDescHeap = DX12Context::Main->CreateSRVHeap();
		ImGui_ImplWin32_Init(*(WindowsWindow::Main->GetHWND()));
		ImGui_ImplDX12_Init(DX12Context::Main->GetDevice(), NUM_FRAMES_IN_FLIGHT,
			DXGI_FORMAT_R8G8B8A8_UNORM, g_pd3dSrvDescHeap,
			g_pd3dSrvDescHeap->GetCPUDescriptorHandleForHeapStart(),
			g_pd3dSrvDescHeap->GetGPUDescriptorHandleForHeapStart());
	}

	void ImGuiLayerDX12::NewFrameBegin()
	{
		SIByL_CORE_TRACE("IMGUI DX12 FRAME BEGIN");
		// Start the Dear ImGui frame
		ImGui_ImplDX12_NewFrame();
		ImGui_ImplWin32_NewFrame();
	}

	void ImGuiLayerDX12::NewFrameEnd()
	{
		ID3D12GraphicsCommandList* g_pd3dCommandList = DX12Context::Main->GetDXGraphicCommandList();
		g_pd3dCommandList->SetDescriptorHeaps(1, &g_pd3dSrvDescHeap);
		ImGui_ImplDX12_RenderDrawData(ImGui::GetDrawData(), g_pd3dCommandList);
	}

	void ImGuiLayerDX12::PlatformDestroy()
	{
		ImGui_ImplDX12_Shutdown();
		ImGui_ImplWin32_Shutdown();
		ImGui::DestroyContext();
	}
}