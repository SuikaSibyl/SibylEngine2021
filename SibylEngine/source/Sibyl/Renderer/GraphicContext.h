#pragma once

#include "Sibyl/Renderer/SwapChain.h"
#include "Sibyl/Renderer/CommandList.h"
#include "Sibyl/Renderer/Synchronizer.h"

namespace SIByL
{
	class GraphicContext
	{
	public:
		virtual void Init() = 0;

	public:
		SwapChain* GetSwapChain() { return m_SwapChain.get(); }
		CommandList* GetCommandList() { return m_CommandList; }
		inline Synchronizer* GetSynchronizer() { return m_Synchronizer.get(); }
		void SetCommandList(CommandList* cmdList) { m_CommandList = cmdList; }

	protected:
		std::unique_ptr<SwapChain> m_SwapChain;
		CommandList* m_CommandList;
		std::unique_ptr<Synchronizer> m_Synchronizer;
	};
}