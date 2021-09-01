#pragma once

namespace SIByL
{
	class SwapChain
	{
	public:
		SwapChain(int width, int height)
			:m_Width(width), m_Height(height) {}

		virtual ~SwapChain() { }

		virtual void BindRenderTarget() {}
		virtual void SetRenderTarget() {}
		virtual void PreparePresent() {}
		virtual void Present() {}
		virtual void Reisze(uint32_t width, uint32_t height) {}

	private:
		int m_Width, m_Height;
	};
}