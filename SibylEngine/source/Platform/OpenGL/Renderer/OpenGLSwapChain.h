#pragma once

#include "SIByLpch.h"
#include "Sibyl/Renderer/SwapChain.h"

namespace SIByL
{
	class OpenGLSwapChain :public SwapChain
	{
	public:
		OpenGLSwapChain();
		OpenGLSwapChain(int width, int height);

		virtual void BindRenderTarget() override;
		virtual void SetRenderTarget() override;
		virtual void PreparePresent() override;
		virtual void Present() override;

	private:
		UINT m_CurrentBackBuffer = 0;
	};
}