#pragma once

#include "Sibyl/ImGui/ImGuiLayer.h"

namespace SIByL
{
	class ImGuiLayerDX12 :public ImGuiLayer
	{
	protected:
		void PlatformInit() override;
		void NewFrameBegin() override;
		void NewFrameEnd() override;
		void PlatformDestroy() override;

	private:
		ID3D12DescriptorHeap* g_pd3dSrvDescHeap = NULL;
	};
}