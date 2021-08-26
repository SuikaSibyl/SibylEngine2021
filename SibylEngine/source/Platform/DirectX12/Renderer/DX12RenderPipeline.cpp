#include "SIByLpch.h"
#include "DX12RenderPipeline.h"

#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/Common/DX12Utility.h"

#include "Sibyl/Core/Layer.h"
#include "Sibyl/Core/Application.h"

#include "Platform/DirectX12/Core/DX12CommandList.h"
#include "Platform/DirectX12/Renderer/DX12SwapChain.h"

#include "Sibyl/ImGui/ImGuiLayer.h"

namespace SIByL
{
	DX12RenderPipeline* DX12RenderPipeline::Main;

	DX12RenderPipeline::DX12RenderPipeline()
	{
		SIByL_CORE_ASSERT(!Main, "DX12 Render Pipeline Already Exists!");
		Main = this;
	}

	void DX12RenderPipeline::DrawFrameImpl()
	{
		DX12GraphicCommandList* cmdList = DX12Context::GetGraphicCommandList();
		SwapChain* swapChain = DX12Context::GetSwapChain();
		DX12Synchronizer* synchronizer = DX12Context::GetSynchronizer();

		synchronizer->StartFrame();
		cmdList->Restart();

		// Bind Swap Chain as Render Target
		// -------------------------------------
		swapChain->SetRenderTarget();

		// Drawcalls
		Application::Get().OnDraw();

		swapChain->PreparePresent();
		cmdList->Execute();

		ImGuiLayer::OnDrawAdditionalWindows();

		swapChain->Present();
		synchronizer->EndFrame();
		//synchronizer->ForceSynchronize();
	}
}