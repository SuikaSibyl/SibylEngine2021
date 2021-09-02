#pragma once

#include "Sibyl/ImGui/ImGuiLayer.h"

namespace SIByL
{
	class ImGuiLayerDX12 :public ImGuiLayer
	{
	public:
		ImGuiLayerDX12();
		~ImGuiLayerDX12();

		struct ImGuiAllocation
		{
			D3D12_CPU_DESCRIPTOR_HANDLE m_CpuHandle;
			D3D12_GPU_DESCRIPTOR_HANDLE m_GpuHandle;
		};

		ImGuiAllocation RegistShaderResource();
		static ImGuiLayerDX12* Get() { return m_Instance; }

	protected:
		virtual void PlatformInit() override;
		virtual void NewFrameBegin() override;
		virtual void DrawCall() override;
		virtual void PlatformDestroy() override;
		virtual void OnDrawAdditionalWindowsImpl() override;

	private:
		ComPtr<ID3D12DescriptorHeap> g_pd3dSrvDescHeap = NULL;
		static ImGuiLayerDX12* m_Instance;
		int m_HeapIndex = 1;
	};
}