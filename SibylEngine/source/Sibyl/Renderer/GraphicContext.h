#pragma once

namespace SIByL
{
	class SwapChain;

	class GraphicContext
	{
	public:
		virtual void Init() = 0;

	public:
		SwapChain* GetSwapChain() { return m_SwapChain.get(); }

	protected:
		std::unique_ptr<SwapChain> m_SwapChain;
	};
}