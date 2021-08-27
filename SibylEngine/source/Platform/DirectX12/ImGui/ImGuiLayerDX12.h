#pragma once

#include "Sibyl/ImGui/ImGuiLayer.h"

namespace SIByL
{
	class ImGuiLayerDX12 :public ImGuiLayer
	{
	public:
		~ImGuiLayerDX12();
	protected:
		virtual void PlatformInit() override;
		virtual void NewFrameBegin() override;
		virtual void NewFrameEnd() override;
		virtual void PlatformDestroy() override;
		virtual void OnDrawAdditionalWindowsImpl() override;

	private:
		ComPtr<ID3D12DescriptorHeap> g_pd3dSrvDescHeap = NULL;
	};
}