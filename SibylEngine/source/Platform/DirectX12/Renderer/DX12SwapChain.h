#pragma once

#include "SIByLpch.h"
#include "Sibyl/Renderer/SwapChain.h"

namespace SIByL
{
	class DX12SwapChain :public SwapChain
	{
	public:
		DX12SwapChain(int width, int height);
		void CreateSwapChain(int width, int height);

	private:
		ComPtr<IDXGISwapChain> m_SwapChain;
	};
}