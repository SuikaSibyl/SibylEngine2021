#pragma once

namespace SIByL
{
	class SwapChain
	{
	public:
		SwapChain(int width, int height)
			:m_Width(width), m_Height(height) {}

		virtual void BindRenderTarget() {}
		virtual void SetRenderTarget() {}
		virtual void PreparePresent() {}
		virtual void Present() {}

	private:
		int m_Width, m_Height;
	};
}