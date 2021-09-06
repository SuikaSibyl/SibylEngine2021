#include "SIByLpch.h"
#include "DX12RenderPipeline.h"

#include "Platform/DirectX12/Common/DX12Context.h"
#include "Platform/DirectX12/Common/DX12Utility.h"

#include "Sibyl/Core/Layer.h"
#include "Sibyl/Core/Application.h"

#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandList.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12CommandQueue.h"
#include "Platform/DirectX12/AbstractAPI/Middle/DX12SwapChain.h"

#include "Sibyl/ImGui/ImGuiLayer.h"


#include "Sibyl/Graphic/AbstractAPI/Core/Top/RenderContextManager.h"
#include "Sibyl/Graphic/AbstractAPI/Core/Top/RenderPipeline.h"

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
		PROFILE_SCOPE_FUNCTION();

		Ref<DX12CommandQueue> cmdQueue = DX12Context::GetSCommandQueue();
		Ref<DX12CommandList> cmdList = cmdQueue->GetCommandList();
		DX12Context::SetInFlightSCmdList(cmdList);

		SwapChain* swapChain = DX12Context::GetSwapChain();

		RenderPipeline* pipeline = RenderPipeline::Get();
		pipeline->Render(RenderContext::GetSRContext(), RenderContext::GetCameras());

		// Bind Swap Chain as Render Target
		// -------------------------------------
		swapChain->SetRenderTarget();

		{
			PROFILE_SCOPE("Draw Layers");
			// Drawcalls
			Application::Get().OnDraw();
			
		}

		swapChain->PreparePresent();

		cmdQueue->ExecuteCommandList(cmdList);

		ImGuiLayer::OnDrawAdditionalWindows();

		swapChain->Present();
	}
}