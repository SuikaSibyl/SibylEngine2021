#pragma once

namespace SIByL
{
	class SwapChain;

	class GraphicContext
	{
	public:
		virtual void Init() = 0;
		virtual void SwipBuffers() = 0;

	protected:
		SwapChain* m_SwapChain;
	};
}